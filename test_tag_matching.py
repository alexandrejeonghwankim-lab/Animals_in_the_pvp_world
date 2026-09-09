"""
Teste la fiabilité du fuzzy matching de tags sur une large gamme de termes
génériques, y compris fautes de frappe, variantes de formatage, et les cas
de bruit historiques découverts au fil des itérations (tags courts qui
matchaient par coïncidence : Bikes, Fox, Gore, Musou, Action, RTS, Flight).

Usage :
  python test_tag_matching.py --data data/games_for_rag.parquet
  python test_tag_matching.py --data data/games_for_rag.parquet --threshold 85
"""

import argparse

import pandas as pd

from fuzzy_tags import build_tag_vocab, expand_tag_fuzzy, DEFAULT_THRESHOLD


TEST_TERMS = {
    "genres": [
        "roguelike", "metroidvania", "souls-like", "battle royale", "visual novel",
        "tower defense", "card game", "deck builder", "platformer", "shooter",
        "rpg", "strategy", "puzzle", "racing", "simulation", "sandbox",
        "city builder", "farming sim", "walking simulator", "fighting game",
    ],
    "mecaniques": [
        "open world", "turn based", "co-op", "multiplayer", "singleplayer",
        "crafting", "survival", "stealth", "procedural generation", "sports",
    ],
    "ambiance_style": [
        "horror", "pixel art", "story rich", "atmospheric", "relaxing",
        "difficult", "casual", "retro", "anime", "colorful",
    ],
    "variantes_et_fautes": [
        "rougelike", "open-world", "coop", "turnbased", "roguelite",
    ],
    "nonsense_faux_positifs": [
        "asdkjqwens", "xyzblah123", "purple monkey dishwasher",
    ],
}

# Faux positifs découverts au fil des itérations précédentes de ce script.
# Chaque tuple (terme, tag_a_ne_pas_matcher) doit rester absent des résultats.
KNOWN_NOISE_PAIRS = [
    ("roguelike", "Bikes"), ("roguelike", "Clicker"),
    ("souls-like", "Bikes"), ("souls-like", "Musou"), ("souls-like", "Clicker"),
    ("sandbox", "Fox"), ("platformer", "Gore"),
    ("sports", "RTS"), ("simulation", "Action"),
    ("procedural generation", "Action"), ("fighting game", "Flight"),
]


def main(data_path: str, threshold: float):
    print(f"Chargement de {data_path}...")
    df = pd.read_parquet(data_path)

    print("Construction du vocabulaire de tags...")
    vocab = build_tag_vocab(df)
    print(f"{len(vocab)} tags uniques trouvés dans le dataset.\n")

    results_by_category = {}

    for category, terms in TEST_TERMS.items():
        print(f"{'=' * 80}")
        print(f"Catégorie : {category}  (seuil = {threshold})")
        print(f"{'=' * 80}\n")

        n_matched = 0
        for term in terms:
            matches = expand_tag_fuzzy(term, vocab, threshold=threshold)
            if matches:
                n_matched += 1
                match_str = ", ".join(f"{tag} ({score:.0f})" for tag, score in matches)
                print(f"  '{term}' ->\n      {match_str}")
            else:
                print(f"  '{term}' -> AUCUN MATCH")

        results_by_category[category] = (n_matched, len(terms))
        print()

    print(f"{'=' * 80}")
    print("VERIFICATION DU BRUIT HISTORIQUE (doit être vide)")
    print(f"{'=' * 80}\n")
    noise_still_present = False
    for term, noisy_tag in KNOWN_NOISE_PAIRS:
        if noisy_tag not in vocab:
            continue  # ce tag précis n'existe pas dans ce dataset, rien à vérifier
        matches = dict(expand_tag_fuzzy(term, vocab, threshold=threshold))
        if noisy_tag in matches:
            print(f"  ENCORE PRESENT : '{term}' matche toujours '{noisy_tag}' "
                  f"(score {matches[noisy_tag]:.0f})")
            noise_still_present = True
    if not noise_still_present:
        print("  Aucun faux positif historique détecté -- le fix tient.")

    print(f"\n{'=' * 80}")
    print("RESUME PAR CATEGORIE")
    print(f"{'=' * 80}")
    for category, (matched, total) in results_by_category.items():
        expectation = "faux positifs attendus à 0" if category == "nonsense_faux_positifs" else ""
        print(f"  {category}: {matched}/{total} matchés  {expectation}")

    print("\nÀ vérifier manuellement en plus des compteurs :")
    print("1. Chaque match affiché a-t-il vraiment un rapport sémantique avec le terme ?")
    print("2. La section 'verification du bruit historique' est-elle bien vide ?")
    print("3. Repère toi-même d'éventuels NOUVEAUX cas de bruit non couverts par")
    print("   KNOWN_NOISE_PAIRS -- ce script ne peut vérifier que ce qu'on lui dit")
    print("   de vérifier, il ne remplace pas une relecture attentive des résultats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/games_for_rag.parquet")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    main(args.data, args.threshold)