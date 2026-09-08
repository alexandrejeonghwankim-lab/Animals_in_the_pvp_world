"""
Préparation des données Steam pour indexation RAG.

Entrée :
  - steam_games.csv
  - steam_game_review.csv (colonnes: app_id, name, reviews)

Sortie :
  - games_for_rag.parquet : un texte embeddable + métadonnées structurées par jeu

Usage:
  python prepare_rag_data.py --games steam_games.csv --reviews steam_game_review.csv --out games_for_rag.parquet
"""

import argparse
import json
import re

import pandas as pd
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Nettoyage de texte
# ---------------------------------------------------------------------------

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # emojis divers
    "\U00002600-\U000027BF"  # symboles/dingbats
    "\U0001F1E6-\U0001F1FF"  # drapeaux
    "]+",
    flags=re.UNICODE,
)


def strip_html(text: str) -> str:
    """Retire les balises HTML d'un texte de description Steam."""
    if not isinstance(text, str) or not text.strip():
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", cleaned).strip()


def strip_emojis(text: str) -> str:
    return EMOJI_PATTERN.sub("", text).strip()


def clean_description(text: str) -> str:
    text = strip_html(text)
    text = strip_emojis(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Parsing des citations presse (steam_game_review.csv)
# ---------------------------------------------------------------------------

# Extraction des citations entre guillemets (gère guillemets typographiques et droits)
QUOTE_ONLY_PATTERN = re.compile(r'[“"]([^“”"]+)[”"]', flags=re.UNICODE)

# Entrées "sans citation" du type "8/10 – Outlet" ou "8 – Outlet", potentiellement répétées
BARE_ENTRY_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:/\s*(?:10|5))?\s*[–-]\s*'
    r'([^–\-\n]+?)(?=\s*\d+(?:\.\d+)?\s*(?:/\s*(?:10|5))?\s*[–-]|$)',
    flags=re.UNICODE,
)


def _parse_trailer(trailer: str) -> tuple[float | None, str | None]:
    """Parse le texte situé après une citation : note optionnelle + nom du média."""
    trailer = trailer.strip()
    if not trailer:
        return None, None
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(?:/\s*(?:10|5))?\s*[–-]?\s*(.*)$', trailer, flags=re.UNICODE)
    if m:
        score = float(m.group(1))
        outlet = m.group(2).strip(' –-') or None
        return score, outlet
    return None, trailer.strip(' –-') or None


def parse_press_reviews(raw: str) -> list[dict]:
    """Extrait une liste de {quote, score, outlet} depuis le texte brut de reviews.

    Gère deux cas :
      - citations entre guillemets, suivies d'une note/média optionnels
        (ex: "quote" 8/10 – Outlet, ou "quote" Outlet sans tiret)
      - entrées sans citation, juste note + média, potentiellement répétées
        (ex: 8/10 – indienova   4/5 – MGR Online)
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    quote_matches = list(QUOTE_ONLY_PATTERN.finditer(raw))
    results = []

    if quote_matches:
        for i, m in enumerate(quote_matches):
            quote = m.group(1).strip()
            start = m.end()
            end = quote_matches[i + 1].start() if i + 1 < len(quote_matches) else len(raw)
            score, outlet = _parse_trailer(raw[start:end])
            results.append({"quote": quote, "score": score, "outlet": outlet})
    else:
        for m in BARE_ENTRY_PATTERN.finditer(raw):
            score = float(m.group(1))
            outlet = m.group(2).strip(' –-') or None
            if outlet:
                results.append({"quote": None, "score": score, "outlet": outlet})

    return results


def press_reviews_to_text(reviews: list[dict]) -> str:
    """Concatène les citations en un bloc de texte lisible pour l'embedding."""
    parts = []
    for r in reviews:
        if r["quote"]:
            parts.append(r["quote"])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def safe_json_list(x):
    """Parse les colonnes stockées en string de liste JSON (genres, tags, categories...).
    Garantit toujours un retour de type list[str], jamais un dict/scalaire/None,
    pour éviter les erreurs de type mixte lors de l'écriture parquet."""
    if isinstance(x, list):
        return [str(v) for v in x]
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        parsed = json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    # Le JSON était valide mais n'était pas une liste (dict, string, nombre...)
    return []


def build_embedding_text(row) -> str:
    parts = [
        row.get("short_description", "") or "",
        row.get("about_the_game_clean", "") or "",
        row.get("press_quotes_text", "") or "",
    ]
    return " ".join(p for p in parts if p).strip()


def main(games_path: str, reviews_path: str, out_path: str):
    print(f"Lecture {games_path}...")
    games = pd.read_csv(games_path, low_memory=False)

    print(f"Lecture {reviews_path}...")
    reviews = pd.read_csv(reviews_path)

    # --- Nettoyage descriptions ---
    print("Nettoyage des descriptions (HTML, emojis)...")
    games["about_the_game_clean"] = games["about_the_game"].apply(clean_description)
    games["short_description"] = games["short_description"].fillna("").apply(
        lambda x: strip_emojis(str(x))
    )

    # --- Parsing des reviews presse ---
    print("Parsing des citations presse...")
    reviews["parsed_reviews"] = reviews["reviews"].apply(parse_press_reviews)
    reviews["press_quotes_text"] = reviews["parsed_reviews"].apply(press_reviews_to_text)
    reviews["press_quote_count"] = reviews["parsed_reviews"].apply(len)
    reviews["press_outlets"] = reviews["parsed_reviews"].apply(
        lambda rs: sorted({r["outlet"] for r in rs if r["outlet"]})
    )
    reviews["press_score_avg"] = reviews["parsed_reviews"].apply(
        lambda rs: (
            round(sum(r["score"] for r in rs if r["score"] is not None) /
                  max(1, len([r for r in rs if r["score"] is not None])), 2)
            if any(r["score"] is not None for r in rs) else None
        )
    )

    # --- Merge ---
    print("Fusion des deux sources...")
    merged = games.merge(
        reviews[["app_id", "press_quotes_text", "press_quote_count", "press_outlets", "press_score_avg"]],
        on="app_id",
        how="left",
    )
    merged["press_quotes_text"] = merged["press_quotes_text"].fillna("")
    merged["press_quote_count"] = merged["press_quote_count"].fillna(0).astype(int)

    # --- Parsing des listes JSON (tags, genres, categories) pour usage en filtre ---
    print("Parsing genres/categories/tags...")
    for col in ["genres", "categories", "tags"]:
        merged[col] = merged[col].apply(safe_json_list)

    # --- Filtrage des jeux invalides ---
    before = len(merged)
    merged = merged[merged["steam_store_available"] == True]  # noqa: E712
    merged = merged[merged["about_the_game_clean"].str.len() > 20]
    print(f"Jeux filtrés : {before} -> {len(merged)}")

    # --- Texte final à embedder ---
    print("Construction du texte à embedder...")
    merged["embedding_text"] = merged.apply(build_embedding_text, axis=1)

    # --- Colonnes finales : texte embeddable + métadonnées pour le payload Qdrant ---
    output_cols = [
        "app_id", "name", "embedding_text",
        "genres", "categories", "tags",
        "price", "price_status", "release_date",
        "metacritic_score", "positive", "negative", "recommendations",
        "average_playtime_forever", "median_playtime_forever",
        "windows", "mac", "linux",
        "press_quote_count", "press_outlets", "press_score_avg",
        "header_image",
    ]
    final = merged[output_cols]

    final.to_parquet(out_path, index=False)
    print(f"Écrit {len(final)} jeux dans {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--out", default="games_for_rag.parquet")
    args = parser.parse_args()
    main(args.games, args.reviews, args.out)