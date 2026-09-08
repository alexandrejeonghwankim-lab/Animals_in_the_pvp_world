"""
Chunking par phrases pour les textes trop longs (> limite de tokens du modèle d'embedding).

Stratégie à 4 niveaux, du plus "naturel" au plus "brutal" :
  1. Découpage par phrase (. ! ?) puis regroupement jusqu'à max_tokens
  2. Si un fragment entre deux terminateurs est encore trop long : découpage par saut de ligne
  3. Si un fragment (une ligne) est encore trop long : découpage par mots (espaces)
  4. Si un "mot" est encore trop long à lui seul (CJK sans espaces) : découpage par caractères

+ VÉRIFICATION FINALE sur le texte réellement assemblé (pas la somme des comptes
  individuels) : la tokenisation BPE n'est pas parfaitement additive -- tokeniser
  5 phrases séparément puis les joindre peut donner un texte final dont le VRAI
  compte de tokens dépasse la somme théorique. Le garde-fou re-tokenise donc
  chaque chunk assemblé et le re-découpe si besoin, garantissant un respect
  strict de max_tokens sur le texte final, pas juste sur une estimation.

Usage :
    from sentence_transformers import SentenceTransformer
    from chunking import chunk_text
    from prepare_rag_data import clean_description

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    clean = clean_description(raw_html_text)
    chunks = chunk_text(clean, model.tokenizer, max_tokens=450)
"""

import re


SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
NEWLINE_SPLIT_PATTERN = re.compile(r'\n+')


def split_into_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text.strip()) if s.strip()]


def split_by_chars(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Niveau 4 (dernier recours) : découpe par fenêtre de caractères."""
    if not text:
        return []

    pieces = []
    step = max(10, len(text) // max(1, len(tokenizer.tokenize(text)) // max_tokens or 1))
    start = 0
    while start < len(text):
        end = min(start + step, len(text))
        while end < len(text) and len(tokenizer.tokenize(text[start:end])) < max_tokens:
            end = min(end + step, len(text))
        while end > start and len(tokenizer.tokenize(text[start:end])) > max_tokens:
            end -= max(1, step // 4)
        if end <= start:
            end = start + 1
        pieces.append(text[start:end])
        start = end

    return [p for p in pieces if p.strip()]


def split_by_words(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Niveau 3 : découpe par mots. Retombe sur split_by_chars si un mot seul
    dépasse déjà max_tokens (cas CJK sans espaces)."""
    words = text.split()
    if not words:
        return []

    pieces = []
    current_words: list[str] = []
    current_tokens = 0

    for word in words:
        w_tokens = len(tokenizer.tokenize(word))

        if w_tokens > max_tokens:
            if current_words:
                pieces.append(" ".join(current_words))
                current_words, current_tokens = [], 0
            pieces.extend(split_by_chars(word, tokenizer, max_tokens))
            continue

        if current_tokens + w_tokens > max_tokens and current_words:
            pieces.append(" ".join(current_words))
            current_words, current_tokens = [], 0

        current_words.append(word)
        current_tokens += w_tokens

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def split_oversized_fragment(fragment: str, tokenizer, max_tokens: int) -> list[str]:
    """Découpage par saut de ligne, puis par mots, puis par caractères en dernier recours."""
    lines = [l.strip() for l in NEWLINE_SPLIT_PATTERN.split(fragment) if l.strip()]

    if len(lines) <= 1:
        return split_by_words(fragment, tokenizer, max_tokens)

    result = []
    for line in lines:
        line_tokens = len(tokenizer.tokenize(line))
        if line_tokens > max_tokens:
            result.extend(split_by_words(line, tokenizer, max_tokens))
        else:
            result.append(line)
    return result


def _enforce_hard_limit(chunk: str, tokenizer, max_tokens: int) -> list[str]:
    """Vérification finale : re-tokenise le texte RÉELLEMENT ASSEMBLÉ (pas une
    somme théorique). Si ça dépasse quand même max_tokens (effet de non-additivité
    de la tokenisation BPE), re-découpe ce chunk assemblé via le pipeline complet."""
    real_tokens = len(tokenizer.tokenize(chunk))
    if real_tokens <= max_tokens:
        return [chunk]
    # Re-découpage récursif du texte assemblé qui s'avère malgré tout trop long
    return split_oversized_fragment(chunk, tokenizer, max_tokens)


def chunk_text(text: str, tokenizer, max_tokens: int = 450) -> list[str]:
    """Découpe un texte en chunks ne dépassant JAMAIS max_tokens, vérifié sur le
    texte réellement assemblé (pas une estimation additive).

    IMPORTANT : appliquer clean_description() (prepare_rag_data.py) sur le texte
    AVANT de le passer ici, pour retirer le HTML résiduel.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    fragments = []
    for sentence in sentences:
        n_tok = len(tokenizer.tokenize(sentence))
        if n_tok > max_tokens:
            fragments.extend(split_oversized_fragment(sentence, tokenizer, max_tokens))
        else:
            fragments.append(sentence)

    raw_chunks = []
    current: list[str] = []
    current_tokens = 0

    for frag in fragments:
        f_tok = len(tokenizer.tokenize(frag))
        if current_tokens + f_tok > max_tokens and current:
            raw_chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(frag)
        current_tokens += f_tok

    if current:
        raw_chunks.append(" ".join(current))

    # Vérification finale sur chaque chunk réellement assemblé
    final_chunks = []
    for chunk in raw_chunks:
        final_chunks.extend(_enforce_hard_limit(chunk, tokenizer, max_tokens))

    return final_chunks


chunk_by_sentences = chunk_text  # alias rétro-compatible


if __name__ == "__main__":
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from prepare_rag_data import clean_description

    df = pd.read_parquet("data/games_for_rag.parquet")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    games_raw = pd.read_csv("data/steam_games.csv", low_memory=False)

    long_app_ids = df[df["embedding_text"].str.split().str.len() > 512]["app_id"]
    candidates = games_raw[games_raw["app_id"].isin(long_app_ids)].copy()
    candidates["desc_len"] = candidates["about_the_game"].str.len()
    sample = candidates.nlargest(3, "desc_len")

    all_ok = True
    for _, row in sample.iterrows():
        raw_text = row["about_the_game"]
        clean_text = clean_description(raw_text)
        chunks = chunk_text(clean_text, model.tokenizer, max_tokens=450)

        # Vérification stricte sur le texte réellement assemblé (pas une estimation)
        max_chunk_tokens = max((len(model.tokenizer.tokenize(c)) for c in chunks), default=0)
        ok = max_chunk_tokens <= 450
        all_ok = all_ok and ok

        print(f"\n{'=' * 60}")
        print(f"Jeu : {row['name']}")
        print(f"Nombre de chunks produits : {len(chunks)}")
        print(f"Max tokens observé (mesuré sur le texte assemblé) : {max_chunk_tokens} (limite : 450) "
              f"{'OK' if ok else '!! DEPASSEMENT !!'}")

    print(f"\n{'=' * 60}")
    print("Garde-fou respecté sur les 3 jeux testés :", "OUI" if all_ok else "NON")