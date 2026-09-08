"""
Indexation complète du dataset Steam dans Qdrant.

- Jeux courts (~93.7%) : un seul point Qdrant, id = app_id * 10000
- Jeux longs (~6.3%) : chunkés (chunk_text), id = app_id * 10000 + chunk_index
- Payload complet dupliqué sur chaque chunk (volume négligeable, cf. discussion)

Prérequis :
  pip install qdrant-client sentence-transformers
  Qdrant doit tourner en local (dans un terminal séparé) :
    docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

Usage :
  # Test rapide sur 500 jeux avant de lancer sur tout le dataset :
  python index_to_qdrant.py --data data/games_for_rag.parquet --limit 500

  # Indexation complète :
  python index_to_qdrant.py --data data/games_for_rag.parquet
"""

import argparse

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer

from chunking import chunk_text

MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768
MAX_TOKENS = 450  # marge de sécurité sous la limite de 512 du modèle
COLLECTION_NAME = "steam_games"


# ---------------------------------------------------------------------------
# Conversion des types pandas/numpy vers des types JSON-sérialisables
# ---------------------------------------------------------------------------

def to_native(value):
    """Convertit un type numpy/pandas en type Python natif pour le payload Qdrant."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if not np.isnan(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, np.ndarray)):
        return [to_native(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, np.ndarray)) else False:
        return None
    return value


def build_payload(row: pd.Series) -> dict:
    """Construit le payload Qdrant à partir d'une ligne du DataFrame.
    Ce payload est identique sur tous les chunks d'un même jeu."""
    fields = [
        "app_id", "name", "genres", "categories", "tags",
        "price", "price_status", "release_date",
        "metacritic_score", "positive", "negative", "recommendations",
        "average_playtime_forever", "median_playtime_forever",
        "windows", "mac", "linux",
        "press_quote_count", "press_outlets", "press_score_avg",
        "header_image",
    ]
    return {f: to_native(row[f]) for f in fields if f in row.index}


# ---------------------------------------------------------------------------
# Construction des points (avec chunking ciblé)
# ---------------------------------------------------------------------------

def build_points_for_game(row: pd.Series, tokenizer, max_tokens: int) -> list[dict]:
    """Retourne une liste de {id, text, payload} pour un jeu.
    Un seul élément si le texte tient dans la limite, plusieurs sinon (chunké)."""
    text = row["embedding_text"]
    app_id = int(row["app_id"])
    payload = build_payload(row)

    n_tokens = len(tokenizer.tokenize(text)) if isinstance(text, str) else 0

    if n_tokens <= max_tokens:
        payload["is_chunk"] = False
        payload["chunk_index"] = 0
        return [{"id": app_id * 10000, "text": text, "payload": payload}]

    # Jeu long -> chunking
    chunks = chunk_text(text, tokenizer, max_tokens=max_tokens)
    points = []
    for i, chunk in enumerate(chunks):
        chunk_payload = dict(payload)  # copie : chaque chunk garde le payload complet
        chunk_payload["is_chunk"] = True
        chunk_payload["chunk_index"] = i
        chunk_payload["chunk_count"] = len(chunks)
        points.append({"id": app_id * 10000 + i, "text": chunk, "payload": chunk_payload})
    return points


# ---------------------------------------------------------------------------
# Setup collection Qdrant
# ---------------------------------------------------------------------------

def setup_collection(client: QdrantClient, name: str, vector_size: int):
    if client.collection_exists(name):
        print(f"Collection '{name}' existe déjà -> suppression et recréation.")
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Index de payload pour un filtrage performant (obligatoire pour filtrer
    # efficacement sur de gros volumes plutôt qu'un scan linéaire)
    keyword_fields = ["genres", "categories", "tags", "windows", "mac", "linux", "is_chunk"]
    for field in keyword_fields:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=PayloadSchemaType.KEYWORD
        )

    float_fields = ["price", "metacritic_score", "average_playtime_forever", "press_score_avg"]
    for field in float_fields:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=PayloadSchemaType.FLOAT
        )

    print(f"Collection '{name}' créée avec index de payload sur {keyword_fields + float_fields}")


# ---------------------------------------------------------------------------
# Indexation par batch
# ---------------------------------------------------------------------------

def index_dataset(df: pd.DataFrame, model: SentenceTransformer, client: QdrantClient,
                   collection_name: str, batch_size: int = 128, max_tokens: int = MAX_TOKENS):
    tokenizer = model.tokenizer

    total_points = 0
    total_chunked_games = 0
    buffer_texts, buffer_ids, buffer_payloads = [], [], []

    def flush():
        nonlocal total_points
        if not buffer_texts:
            return
        vectors = model.encode(
            buffer_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        points = [
            PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            for pid, vec, payload in zip(buffer_ids, vectors, buffer_payloads)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=False)
        total_points += len(points)
        buffer_texts.clear()
        buffer_ids.clear()
        buffer_payloads.clear()

    n_games = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        game_points = build_points_for_game(row, tokenizer, max_tokens)
        if len(game_points) > 1:
            total_chunked_games += 1

        for point in game_points:
            buffer_texts.append(point["text"])
            buffer_ids.append(point["id"])
            buffer_payloads.append(point["payload"])

        if len(buffer_texts) >= batch_size:
            flush()

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n_games} jeux traités, {total_points} points indexés jusqu'ici...")

    flush()  # reste du buffer
    print(f"\nIndexation terminée :")
    print(f"  {n_games} jeux traités")
    print(f"  {total_points} points au total")
    print(f"  {total_chunked_games} jeux ont été chunkés ({total_chunked_games / n_games * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str, qdrant_url: str, batch_size: int, limit: int | None):
    print(f"Chargement de {data_path}...")
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["embedding_text"])
    df = df[df["embedding_text"].str.len() > 0]

    if limit:
        print(f"Mode test : limite à {limit} jeux (échantillon aléatoire reproductible).")
        df = df.sample(n=min(limit, len(df)), random_state=42)

    print(f"Chargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Device : {model.device}")

    client = QdrantClient(url=qdrant_url)
    setup_collection(client, COLLECTION_NAME, VECTOR_SIZE)

    print(f"Indexation de {len(df)} jeux...")
    index_dataset(df, model, client, COLLECTION_NAME, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/games_for_rag.parquet")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None,
                         help="Limite le nombre de jeux indexés (pour tester avant l'indexation complète)")
    args = parser.parse_args()
    main(args.data, args.qdrant_url, args.batch_size, args.limit)