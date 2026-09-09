"""
Interface Chainlit pour l'agent de recommandation de jeux Steam.

Tout ce qui est visible par l'utilisateur final est en anglais (messages
Chainlit, réponses de l'agent). Les commentaires restent en français pour
la maintenance du code.

Prérequis :
  pip install chainlit
  (les autres dépendances -- groq, qdrant-client, sentence-transformers,
  python-dotenv, rapidfuzz -- sont déjà nécessaires depuis les étapes
  précédentes du projet)

Usage :
  chainlit run app.py -w
"""

import os

import chainlit as cl
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from agent import GameAgent
from fuzzy_tags import build_tag_vocab

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY manquante -- vérifie ton fichier .env avant de lancer Chainlit."
    )

# ---------------------------------------------------------------------------
# Ressources lourdes chargées UNE SEULE FOIS au démarrage de l'app (pas par
# session utilisateur) : recharger le modèle d'embedding à chaque connexion
# serait beaucoup trop lent. Seul l'historique de conversation est propre à
# chaque session (géré via cl.user_session, voir on_chat_start plus bas).
# ---------------------------------------------------------------------------
print("Chargement des ressources partagées (une seule fois)...")
_df = pd.read_parquet("data/games_for_rag.parquet")
VOCAB = build_tag_vocab(_df)
EMBED_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
QDRANT_CLIENT = QdrantClient(url="http://localhost:6333", timeout=30)
print("Ressources chargées, l'app est prête.")


@cl.on_chat_start
async def start():
    """Appelé une fois par nouvelle session de chat. Crée un agent dédié
    (avec son propre historique de conversation) pour cet utilisateur --
    les ressources lourdes (modèle, vocab, client Qdrant) sont partagées et
    réutilisées depuis les variables globales ci-dessus."""
    groq_client = Groq(api_key=GROQ_API_KEY)
    agent = GameAgent(groq_client, QDRANT_CLIENT, EMBED_MODEL, VOCAB)
    cl.user_session.set("agent", agent)

    await cl.Message(
        content=(
            "Hi! Tell me what kind of game you're looking for -- genre, "
            "mood, budget, platform -- and I'll suggest some Steam titles "
            "for you. You can also ask follow-up questions about the games "
            "I recommend."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Appelé à chaque message envoyé par l'utilisateur dans une session."""
    agent: GameAgent = cl.user_session.get("agent")

    # agent.ask() est bloquant (appels réseau Groq + Qdrant) ; cl.make_async
    # l'exécute dans un thread séparé pour ne pas geler l'interface pendant
    # ce temps.
    answer = await cl.make_async(agent.ask)(message.content)

    await cl.Message(content=answer).send()