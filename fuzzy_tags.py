"""
Fuzzy matching de tags : résout un terme générique tapé par l'utilisateur
(ex: "roguelike", "open world", "coop") vers les vrais tags Steam présents
dans le dataset (ex: "Action Roguelike", "Open World Survival Craft"), sans
dictionnaire statique à maintenir manuellement.

Historique des corrections (issues de tests empiriques, voir test_tag_matching.py) :

  1. Bug de casse : rapidfuzz.fuzz.partial_ratio est sensible à la casse par
     défaut ("rpg" vs "RPG" -> 0.0 sans normalisation, 100 avec).
     -> Fix : lowercase systématique avant comparaison.

  2. Bruit sur variantes de formatage : partial_ratio brut donnait des scores
     élevés à des tags courts sans rapport ("Fox" ~ "sandbox" = 80) par pure
     coïncidence de lettres.
     -> Fix : suppression des espaces/tirets avant comparaison, ce qui aligne
     les vraies variantes en sous-chaînes quasi-exactes (score ~100) et crée
     un écart net avec le bruit restant (~80).

  3. Bruit résiduel sur tags COURTS : même après le fix 2, des tags courts à
     mot unique continuent de matcher par coïncidence des termes sans rapport
     ("Action" ~ "simulation"/"procedural generation" = 91, "RTS" ~ "sports"
     = 100 car "sports" contient littéralement "rts", "Flight" ~ "fighting
     game" = 91). Le point commun : le plus court des deux termes comparés
     fait <= 6 caractères, où la probabilité de collision par hasard devient
     trop élevée pour qu'un score flou soit fiable.
     -> Fix : quand le plus court des deux termes normalisés fait <= 6
     caractères, on exige une égalité EXACTE (pas de score flou du tout).
     Le fuzzy matching (typos, variantes) reste actif uniquement au-delà de
     cette longueur, où le risque de coïncidence redevient négligeable.

Limite connue et acceptée : ce fix 3 élimine le bruit mais peut aussi rater
de vrais rapprochements sur des tags courts qui ne sont pas des sous-chaînes
exactes (ex: "pixel art" vs "Pixel Graphics" reste à un score de 80, sous le
seuil, car les deux mots après "pixel" diffèrent). Compromis accepté : mieux
vaut un faux négatif occasionnel qu'un filtre qui inclut silencieusement des
jeux sans rapport.

Usage :
    from fuzzy_tags import build_tag_vocab, expand_tag_fuzzy

    vocab = build_tag_vocab(df)  # une fois, au démarrage
    matches = expand_tag_fuzzy("roguelike", vocab, threshold=88)
"""

import re

from rapidfuzz import fuzz

DEFAULT_THRESHOLD = 88.0
SHORT_TERM_CUTOFF = 6  # en dessous de cette longueur (normalisée), exact-match uniquement

_SEP_PATTERN = re.compile(r'[\s\-]+')


def _normalize(s: str) -> str:
    """Lowercase + suppression des espaces/tirets."""
    return _SEP_PATTERN.sub('', s.lower())


def _score(normalized_term: str, normalized_tag: str, threshold: float) -> float:
    """Calcule le score de matching entre un terme et un tag déjà normalisés.
    Retourne 0 (rejet) si le plus court des deux fait <= SHORT_TERM_CUTOFF
    caractères et qu'il ne s'agit pas d'une égalité exacte -- voir le point 3
    de l'historique en haut du fichier."""
    if normalized_term == normalized_tag:
        return 100.0

    shorter_len = min(len(normalized_term), len(normalized_tag))
    if shorter_len <= SHORT_TERM_CUTOFF:
        return 0.0

    return fuzz.partial_ratio(normalized_term, normalized_tag)


def build_tag_vocab(df) -> list[str]:
    """Extrait la liste unique de tous les tags présents dans le dataset."""
    all_tags = set()
    for tags in df["tags"]:
        all_tags.update(tags)
    return sorted(all_tags)


def expand_tag_fuzzy(
    term: str,
    vocab: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = 6,
) -> list[tuple[str, float]]:
    """Trouve les tags réels du vocabulaire les plus proches du terme donné.

    Retourne une liste de (tag_original, score) triée par score décroissant,
    uniquement les matchs au-dessus de threshold (0-100).
    """
    if not term or not vocab:
        return []

    normalized_term = _normalize(term)

    scored = []
    for tag in vocab:
        score = _score(normalized_term, _normalize(tag), threshold)
        if score >= threshold:
            scored.append((tag, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def expand_tag_fuzzy_names_only(
    term: str, vocab: list[str], threshold: float = DEFAULT_THRESHOLD
) -> list[str]:
    """Comme expand_tag_fuzzy, mais retourne juste les noms de tags (pour usage
    direct dans un filtre Qdrant MatchAny)."""
    return [tag for tag, _score in expand_tag_fuzzy(term, vocab, threshold)]