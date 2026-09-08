"""
Comparaison de deux modèles d'embedding sur des paires de jeux Steam.

Usage:
  python compare_embedding_models.py --data data/games_for_rag.parquet
"""

import argparse
import time

import pandas as pd
from sentence_transformers import SentenceTransformer, util


MODELS_TO_COMPARE = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]


def pick_game_pairs(df: pd.DataFrame) -> dict:
    """Sélectionne quelques paires de jeux pour comparer les modèles.

    Adapte les noms ci-dessous à ce qui existe réellement dans ton dataset
    (fais une recherche par mot-clé dans df['name'] si besoin).
    """
    def find(name_substring: str):
        matches = df[df["name"].str.contains(name_substring, case=False, na=False)]
        if matches.empty:
            return None
        return matches.iloc[0]

    pairs = {}

    # Exemple : deux jeux thématiquement proches (à adapter selon ton dataset)
    similar_pair = ("Among Trees", "Bladequest")  # remplace par 2 jeux similaires trouvés chez toi
    a, b = find(similar_pair[0]), find(similar_pair[1])
    if a is not None and b is not None:
        pairs["similar"] = (a, b)

    # Exemple : deux jeux très différents
    different_pair = ("Among Trees", "Gun Crazy")  # remplace par 2 jeux différents
    a, b = find(different_pair[0]), find(different_pair[1])
    if a is not None and b is not None:
        pairs["different"] = (a, b)

    return pairs


def longest_text_row(df: pd.DataFrame):
    """Retourne le jeu avec le texte le plus long, pour tester la troncature."""
    idx = df["embedding_text"].str.len().idxmax()
    return df.loc[idx]


def evaluate_model(model_name: str, pairs: dict, long_text_row: pd.Series):
    print(f"\n{'=' * 60}")
    print(f"Modèle : {model_name}")
    print("=" * 60)

    load_start = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_time = time.perf_counter() - load_start
    print(f"Device utilisé : {model.device}")
    print(f"Temps de chargement : {load_time:.2f}s")
    print(f"Dimension du vecteur : {model.get_sentence_embedding_dimension()}")
    print(f"Longueur de séquence max (tokens) : {model.max_seq_length}")

    # --- Similarité sur les paires ---
    for label, (game_a, game_b) in pairs.items():
        text_a = game_a["embedding_text"]
        text_b = game_b["embedding_text"]

        enc_start = time.perf_counter()
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        enc_time = time.perf_counter() - enc_start

        score = util.cos_sim(emb_a, emb_b).item()
        print(f"\n  Paire '{label}': {game_a['name']} <-> {game_b['name']}")
        print(f"    Similarité cosinus : {score:.4f}")
        print(f"    Temps d'encodage des 2 textes : {enc_time:.3f}s")

    # --- Test sur le texte le plus long (troncature) ---
    text = long_text_row["embedding_text"]
    n_chars = len(text)
    n_tokens_est = len(model.tokenizer.tokenize(text))
    print(f"\n  Texte le plus long du dataset : '{long_text_row['name']}'")
    print(f"    Longueur : {n_chars} caractères, ~{n_tokens_est} tokens")
    print(f"    max_seq_length du modèle : {model.max_seq_length} tokens")
    if n_tokens_est > model.max_seq_length:
        print(f"    -> TRONCATURE : ~{n_tokens_est - model.max_seq_length} tokens seront ignorés")
    else:
        print("    -> Pas de troncature pour ce texte")


def main(data_path: str):
    print(f"Chargement de {data_path}...")
    df = pd.read_parquet(data_path)

    pairs = pick_game_pairs(df)
    if not pairs:
        print("ATTENTION : aucune paire de jeux trouvée avec les noms par défaut.")
        print("Édite pick_game_pairs() avec des noms de jeux qui existent dans ton dataset.")
        return

    long_row = longest_text_row(df)

    for model_name in MODELS_TO_COMPARE:
        evaluate_model(model_name, pairs, long_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/games_for_rag.parquet")
    args = parser.parse_args()
    main(args.data)