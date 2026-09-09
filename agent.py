"""
Agent conversationnel : transforme une requête en langage naturel en appel
structuré à search_games(), via function calling (Groq / openai/gpt-oss-120b).

Tout ce qui est visible par l'utilisateur final (system prompt, schéma
d'outil, messages retournés au chat) est en anglais. Les commentaires et
docstrings restent en français pour la maintenance du code.

Prérequis :
  pip install groq python-dotenv
  Variable d'environnement GROQ_API_KEY (clé gratuite sur console.groq.com)

Usage :
  python agent.py
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from fuzzy_tags import build_tag_vocab
from grounding_check import check_grounding, format_grounding_warning
from retrieval import search_games

load_dotenv()  # charge les variables du fichier .env dans os.environ

MODEL_NAME = "openai/gpt-oss-120b"
TOP_K = 5  # fixé par le code, pas laissé au choix du LLM

SYSTEM_PROMPT = """You are an assistant that helps users find Steam games.

You have access to a search tool that combines semantic search (on the
game's description and mood) with structured filters (tags, price,
Metacritic score, platform).

When the user describes what they're looking for:
- Extract explicit structured constraints (genre/mechanic tags, maximum
  budget, platform) into the filter parameters.
- Use query_text for anything related to feeling, atmosphere, style, or
  comparison to other games (e.g. "like Hollow Knight but shorter",
  "dark and melancholic atmosphere").
- Never invent an exact tag: pass the term as the user phrased it (e.g.
  "roguelike"), the system resolves it to the real tags on its own.
- If the user gives no structured constraint at all, still call the tool
  with just query_text.

CRITICAL -- grounding rule: once you receive the tool results, you may
ONLY state facts (price, whether it's free, tags/mechanics like PvP or
Co-op, Metacritic score, playtime) that are LITERALLY present in that tool
result for that specific game. Do not fill in gaps using your own general
knowledge of these games, even if you recognize the title and believe you
know more about it -- the tool result is your only source of truth. If a
detail isn't in the tool result, don't mention it rather than guess. This
applies even to well-known games: only report what the data actually says.

Always answer in English, regardless of the language the user writes in.

After receiving the results, present them naturally and concisely,
briefly explaining why each game fits the request. Don't just list the raw
data -- give a real recommendation.
"""

SEARCH_GAMES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_games",
        "description": (
            "Search Steam games by semantic description and optional "
            "structured filters (tags, price, critic score, platform)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": (
                        "Natural language description of the desired mood, "
                        "style, or gameplay. Always provide this field."
                    ),
                },
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": (
                        "Explicitly requested genres or mechanics (e.g. "
                        "'roguelike', 'open world', 'co-op'). Generic terms "
                        "are accepted as-is, no need to know the exact "
                        "Steam tags. Omit or use null if not specified."
                    ),
                },
                "max_price": {
                    "type": ["number", "null"],
                    "description": "Maximum budget in euros, if mentioned. Use null if not specified.",
                },
                "min_metacritic": {
                    "type": ["number", "null"],
                    "description": "Minimum Metacritic score, if a quality bar is requested. Use null if not specified.",
                },
                "platforms": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["windows", "mac", "linux"]},
                    "description": "Required platforms, if mentioned. Use null if not specified.",
                },
            },
            "required": ["query_text"],
        },
    },
}


class GameAgent:
    def __init__(self, groq_client, qdrant_client, embed_model, tag_vocab):
        self.groq = groq_client
        self.qdrant = qdrant_client
        self.embed_model = embed_model
        self.vocab = tag_vocab
        # Historique conservé en interne -> chaque instance = une conversation.
        # Pour un multi-utilisateur (Chainlit), on crée une instance par session.
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _execute_search(self, arguments: dict) -> list[dict]:
        """Appelle search_games() avec les arguments extraits par le LLM,
        et sérialise les résultats pour les renvoyer au modèle."""
        results = search_games(
            self.qdrant, self.embed_model, self.vocab,
            query_text=arguments.get("query_text", ""),
            tags=arguments.get("tags"),
            max_price=arguments.get("max_price"),
            min_metacritic=arguments.get("min_metacritic"),
            platforms=arguments.get("platforms"),
            top_k=TOP_K,
        )
        return [
            {
                "name": r.payload["name"],
                "price": r.payload.get("price"),
                "tags": r.payload.get("tags", [])[:6],
                "metacritic_score": r.payload.get("metacritic_score"),
                "average_playtime_hours": (
                    round(r.payload["average_playtime_forever"] / 60, 1)
                    if r.payload.get("average_playtime_forever") else None
                ),
                "relevance_score": round(r.score, 3),
            }
            for r in results
        ]

    def ask(self, user_message: str, max_rounds: int = 5) -> str:
        """Boucle d'appels d'outils : tant que le LLM demande à utiliser
        search_games, on exécute et on lui renvoie les résultats, jusqu'à ce
        qu'il produise une réponse finale sans appel d'outil. L'historique de
        la conversation (self.messages) persiste entre les appels, ce qui
        permet des questions de suivi sans tout redonner comme contexte."""
        self.messages.append({"role": "user", "content": user_message})
        all_search_results: list[dict] = []  # accumulé sur ce tour pour le grounding check

        for _ in range(max_rounds):
            response = self.groq.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                tools=[SEARCH_GAMES_TOOL],
                tool_choice="auto",
                max_completion_tokens=1024,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                content = message.content
                if not content or not content.strip():
                    content = getattr(message, "reasoning", None)
                content = content or "(empty response received from the model -- please try again)"

                self.messages.append({"role": "assistant", "content": content})

                if all_search_results:
                    report = check_grounding(content, all_search_results)
                    warning = format_grounding_warning(report)
                    if warning:
                        print(warning)

                return content

            # Le message assistant est reconstruit manuellement (pas de
            # model_dump() direct, cf. bug "annotations" rencontré plus tôt)
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                print(f"  [agent calling search_games with: {arguments}]")
                search_results = self._execute_search(arguments)
                all_search_results.extend(search_results)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(search_results, ensure_ascii=False),
                })

        fallback = "(the agent could not reach a conclusion after several attempts -- please rephrase your request)"
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback


def main():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "Variable d'environnement GROQ_API_KEY manquante. "
            "Récupère une clé gratuite sur console.groq.com et définis-la "
            "avant de lancer ce script."
        )

    print("Chargement du vocabulaire de tags...")
    df = pd.read_parquet("data/games_for_rag.parquet")
    vocab = build_tag_vocab(df)

    print("Chargement du modèle d'embedding...")
    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    qdrant = QdrantClient(url="http://localhost:6333", timeout=30)
    groq_client = Groq(api_key=groq_api_key)

    test_queries = [
        "a cheap roguelike with a dark and oppressive atmosphere",
        "I'm looking for a relaxing nature exploration game",
    ]

    for query in test_queries:
        # Un agent frais par requête ici : ce sont deux essais indépendants,
        # pas une vraie conversation. Dans Chainlit, une instance = une vraie
        # session utilisateur, et l'historique doit persister entre les tours.
        agent = GameAgent(groq_client, qdrant, embed_model, vocab)
        print(f"\n{'=' * 70}")
        print(f"User: {query}")
        print(f"{'=' * 70}")
        answer = agent.ask(query)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()