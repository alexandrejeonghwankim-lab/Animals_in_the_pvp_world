# Steam Games RAG Assistant

A hybrid RAG (Retrieval-Augmented Generation) chatbot that helps users find Steam games through natural language queries. Combines semantic search over game descriptions with structured filtering (tags, price, platform, Metacritic score), wrapped in a function-calling agent with a conversational UI.

Built as part of a team project (data engineering / data science / RAG+agent) at BeCode.

## What it does

Ask things like:
- *"a relaxing exploration game about nature and survival"*
- *"a cheap roguelike with a dark atmosphere, works on Linux"*
- *"a highly rated RPG, at least 80 on Metacritic"*

...and get grounded recommendations pulled from a ~130k-game Steam dataset, with follow-up questions supported ("which one has the best price?").

## Architecture

```
User query
   |
   v
Agent (Groq / openai/gpt-oss-120b, function calling)
   |
   |--- extracts: query_text (semantic) + tags/price/platform/metacritic (structured)
   v
search_games()
   |
   |--- semantic: embed query_text -> vector search (Qdrant, cosine)
   |--- structured: fuzzy-resolve tags -> Qdrant payload filter
   |--- oversampled query (top_k x 3) to survive deduplication
   v
Deduplicate by game (chunked games can return multiple hits)
   v
Results -> back to agent -> natural language answer
   |
   v
Grounding check (diagnostic): flags cited facts not present in results
```

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embedding model | `BAAI/bge-base-en-v1.5` | Better token coverage (512) than MiniLM (256) for long game descriptions; empirically better discrimination on domain pairs |
| Vector DB | Qdrant (local, Docker) | Native payload filtering combined with vector search in one query |
| LLM / agent | Groq API, `openai/gpt-oss-120b` | Free tier, fast inference, OpenAI-compatible function calling |
| Tag resolution | `rapidfuzz` (fuzzy matching) | No static synonym dictionary to maintain; resolves generic terms ("roguelike") to real Steam tags ("Action Roguelike", "Rogue-lite"...) |
| UI | Chainlit | Simple chat UI with per-session state for multi-turn conversations |

## Project structure

```
prepare_rag_data.py     Cleans raw CSVs (HTML stripping, press-quote parsing),
                         builds the embeddable text + metadata parquet
chunking.py              Splits long game descriptions into token-safe chunks
                         (sentence -> line -> word -> character fallback cascade)
index_to_qdrant.py       Embeds and indexes everything into Qdrant, chunking
                         only the ~6% of games that need it
retrieval.py             Hybrid search: semantic + structured filter + dedup
fuzzy_tags.py            Resolves generic tag terms to real dataset tags
grounding_check.py       Diagnostic: detects hallucinated game names and facts
                         (price, PvP/Co-op...) in the agent's final answer
agent.py                 GameAgent class: function-calling loop against
                         search_games(), stateful conversation history
app.py                   Chainlit entry point (chat UI)
test_tag_matching.py     Reliability test suite for fuzzy_tags.py
```

## Setup

```bash
pip install pandas beautifulsoup4 pyarrow sentence-transformers qdrant-client \
            rapidfuzz groq python-dotenv chainlit

# Start Qdrant (keep running in a separate terminal)
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# .env file with:
# GROQ_API_KEY=your_key_here   (free key at console.groq.com)
```

Pipeline, run once:
```bash
python prepare_rag_data.py --games data/steam_games.csv --reviews data/steam_games_reviews.csv --out data/games_for_rag.parquet
python index_to_qdrant.py --data data/games_for_rag.parquet
```

Run the app:
```bash
python -m chainlit run app.py -w
```

## Key design decisions

**Chunking only where needed.** Word-count estimates suggested ~6% of games needed chunking (text exceeding the embedding model's 512-token limit); the real token-based figure turned out closer to 20%, since word count underestimates true BPE token count, especially for non-English text. Chunking uses a 4-level fallback (sentence -> newline -> word -> character) to guarantee no chunk ever exceeds the token limit, regardless of language or text structure -- this was only reached after finding that a single-level regex silently produced 25,000+ token "chunks" on text with no sentence punctuation.

**Tags are filters, not embedded text.** Genres/tags/categories are structured, low-cardinality data -- they belong in Qdrant payload filters, not in the text passed to the embedding model. This keeps semantic search focused on genuinely free-form text (descriptions, mood) and makes filtering exact rather than approximate.

**Fuzzy tag matching over a static synonym dictionary.** A hand-maintained dictionary doesn't scale to arbitrary user phrasing. `fuzzy_tags.py` resolves generic terms against the real tag vocabulary at query time. This went through three rounds of empirical bug-fixing: a case-sensitivity bug in the underlying scorer, coincidental substring matches on formatting variants (fixed by comparing separator-stripped strings), and coincidental substring matches on short tags specifically (fixed by requiring exact equality when the shorter compared string is <=6 characters). Each fix was validated against a reproducible test suite (`test_tag_matching.py`) covering genres, mechanics, typos, and deliberately nonsensical input.

**Grounding is checked at two levels.** A cited game name existing in the search results doesn't guarantee the *facts* stated about it are accurate -- an LLM can attach a real game's name to an invented price or mechanic. `grounding_check.py` checks both: does the cited name exist in the results, and do facts stated near that name (price, PvP/Co-op/Multiplayer claims) match what the tool actually returned. Notably, this checks fidelity to the *provided context*, not objective truth -- a model asserting something true in reality but absent from the retrieved data is still flagged, since that indicates reliance on general training knowledge rather than the retrieved context.

**Stateful agent, one instance per session.** The agent keeps its full message history internally rather than rebuilding it per call, enabling follow-up questions. Heavy resources (embedding model, tag vocabulary, Qdrant client) are loaded once at app startup and shared across sessions; only the lightweight agent instance (message history) is created per Chainlit session.

## Known limitations

- Fuzzy tag matching trades recall for precision on short terms: `"pixel art"` doesn't resolve to `"Pixel Graphics"` (score just below threshold) to avoid reopening false-positive matches on short tags.
- The grounding check only validates price and a fixed set of mechanic keywords (PvP, Co-op, Multiplayer, Singleplayer) -- it doesn't verify Metacritic scores, playtime, or free-text claims, though manual spot-checks across multiple test queries found no discrepancies there either.
- The grounding check is diagnostic only (prints a console warning); it doesn't yet trigger an automatic correction round with the LLM.
- No automated evaluation suite yet (retrieval precision/recall, faithfulness scoring) -- validation so far has been manual, query-by-query.

## Possible next steps

- Auto-correction: when `grounding_check` flags an issue, feed it back to the LLM for a corrected answer before showing it to the user.
- A small labeled query/expected-result set to measure retrieval quality (precision@k) rather than spot-checking individual queries.
- Extend `MECHANIC_KEYWORDS` in `grounding_check.py` and `TAG_SYNONYMS`-equivalent coverage as more query patterns are observed in real use.