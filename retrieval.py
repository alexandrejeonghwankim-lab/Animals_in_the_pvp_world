"""
Retrieval hybride : filtrage structuré (tags, prix, plateforme...) combiné à la
recherche sémantique, avec dédoublonnage des jeux chunkés et résolution des
tags génériques via fuzzy matching (voir fuzzy_tags.py).

Usage :
    import pandas as pd
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from fuzzy_tags import build_tag_vocab
    from retrieval import search_games

    df = pd.read_parquet("data/games_for_rag.parquet")
    vocab = build_tag_vocab(df)  # une seule fois au démarrage, pas à chaque requête

    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    results = search_games(
        client, model, vocab,
        query_text="a relaxing exploration game about nature",
        tags=["survival"],
        max_price=20.0,
        top_k=10,
    )
"""

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    Range,
)

from fuzzy_tags import expand_tag_fuzzy_names_only, DEFAULT_THRESHOLD

COLLECTION_NAME = "steam_games"


# ---------------------------------------------------------------------------
# Construction du filtre structuré
# ---------------------------------------------------------------------------

def build_filter(
    vocab: list[str],
    tags: list[str] | None = None,
    genres: list[str] | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_metacritic: float | None = None,
    platforms: list[str] | None = None,  # ex: ["windows"], ["mac", "linux"]
    tag_match_threshold: float = DEFAULT_THRESHOLD,
) -> Filter | None:
    """Construit un filtre Qdrant à partir de contraintes structurées.

    Les tags sont résolus vers leurs vraies variantes dans le dataset via
    fuzzy matching (fuzzy_tags.expand_tag_fuzzy_names_only) plutôt qu'un
    dictionnaire statique -- fonctionne sur n'importe quel terme générique,
    pas seulement ceux explicitement mappés à l'avance.

    Retourne None si aucune contrainte n'est fournie (recherche sémantique pure).
    """
    conditions = []

    if tags:
        expanded = []
        for t in tags:
            resolved = expand_tag_fuzzy_names_only(t, vocab, threshold=tag_match_threshold)
            expanded.extend(resolved)
        if expanded:
            # Dédoublonne (plusieurs termes utilisateur peuvent résoudre vers le même tag)
            expanded = list(dict.fromkeys(expanded))
            conditions.append(FieldCondition(key="tags", match=MatchAny(any=expanded)))
        # Si aucun tag ne matche, on n'ajoute pas de condition tags -> on ne
        # filtre pas plutôt que de générer un filtre vide qui exclurait tout.

    if genres:
        conditions.append(FieldCondition(key="genres", match=MatchAny(any=genres)))

    if min_price is not None or max_price is not None:
        conditions.append(
            FieldCondition(key="price", range=Range(gte=min_price, lte=max_price))
        )

    if min_metacritic is not None:
        conditions.append(
            FieldCondition(key="metacritic_score", range=Range(gte=min_metacritic))
        )

    if platforms:
        for platform in platforms:
            conditions.append(FieldCondition(key=platform, match=MatchValue(value=True)))

    if not conditions:
        return None

    return Filter(must=conditions)


# ---------------------------------------------------------------------------
# Dédoublonnage des résultats (jeux chunkés = plusieurs points)
# ---------------------------------------------------------------------------

def deduplicate_by_game(scored_points: list, top_k: int) -> list:
    """Regroupe les points par app_id, garde le meilleur score par jeu."""
    best_per_game: dict[int, object] = {}

    for point in scored_points:
        app_id = point.payload["app_id"]
        if app_id not in best_per_game or point.score > best_per_game[app_id].score:
            best_per_game[app_id] = point

    deduped = sorted(best_per_game.values(), key=lambda p: p.score, reverse=True)
    return deduped[:top_k]


# ---------------------------------------------------------------------------
# Recherche complète (sémantique + filtre + dédoublonnage)
# ---------------------------------------------------------------------------

def search_games(
    client,
    model,
    vocab: list[str],
    query_text: str,
    top_k: int = 10,
    oversample_factor: int = 3,
    tags: list[str] | None = None,
    genres: list[str] | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_metacritic: float | None = None,
    platforms: list[str] | None = None,
) -> list:
    """Recherche hybride complète. Retourne une liste de points Qdrant dédupliqués
    par jeu (un seul résultat par app_id), triés par score décroissant.

    vocab : liste des tags réels du dataset, construite une fois via
    fuzzy_tags.build_tag_vocab(df) et réutilisée entre les appels (ne pas
    la reconstruire à chaque requête).
    """
    query_filter = build_filter(
        vocab, tags=tags, genres=genres, min_price=min_price, max_price=max_price,
        min_metacritic=min_metacritic, platforms=platforms,
    )

    query_vector = model.encode(query_text).tolist()

    raw_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k * oversample_factor,
    ).points

    return deduplicate_by_game(raw_results, top_k)


if __name__ == "__main__":
    import pandas as pd
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from fuzzy_tags import build_tag_vocab

    print("Chargement du vocabulaire de tags...")
    df = pd.read_parquet("data/games_for_rag.parquet")
    vocab = build_tag_vocab(df)
    print(f"{len(vocab)} tags chargés.\n")

    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    print("=" * 60)
    print("Test 1 : recherche sémantique pure")
    print("=" * 60)
    results = search_games(
        client, model, vocab,
        query_text="a relaxing exploration game about nature and survival",
        top_k=10,
    )
    for r in results:
        print(f"{r.score:.3f} - {r.payload['name']} (chunk={r.payload['is_chunk']})")

    print("\n" + "=" * 60)
    print("Test 2 : recherche hybride avec filtres (tags fuzzy + prix)")
    print("=" * 60)
    results2 = search_games(
        client, model, vocab,
        query_text="dark atmospheric dungeon crawler",
        tags=["roguelike"],
        max_price=15.0,
        top_k=10,
    )
    if not results2:
        print("Aucun résultat -> vérifie que le filtre n'est pas trop restrictif.")
    for r in results2:
        print(f"{r.score:.3f} - {r.payload['name']} - {r.payload['price']}€ "
              f"- tags: {r.payload['tags'][:5]}")

    print("\n" + "=" * 60)
    print("Test 3 : terme avec faute de frappe ('rougelike')")
    print("=" * 60)
    results3 = search_games(
        client, model, vocab,
        query_text="dark atmospheric dungeon crawler",
        tags=["rougelike"],  # faute volontaire
        max_price=15.0,
        top_k=5,
    )
    if not results3:
        print("Aucun résultat -> la faute de frappe n'a pas été résolue correctement.")
    for r in results3:
        print(f"{r.score:.3f} - {r.payload['name']} - {r.payload['price']}€")