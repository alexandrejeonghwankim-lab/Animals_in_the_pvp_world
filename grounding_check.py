"""
Vérifie que la réponse finale de l'agent est bien ancrée dans les résultats
réels de search_games(), à deux niveaux :

  1. Niveau NOM : les jeux cités existent-ils bien dans les résultats reçus ?
     (détecte les jeux purement inventés)
  2. Niveau FAIT : pour un jeu qui existe bien, les détails cités à côté de
     son nom (prix, gratuité, mécaniques comme PvP/Co-op) correspondent-ils
     aux vraies données ? (détecte les attributs inventés sur un vrai jeu --
     ex: un jeu à 29,99€ décrit comme "free to play")

Le niveau 2 a été ajouté après un cas réel observé : l'agent a cité un jeu
existant (Fireteam) mais lui a attribué un prix et un tag PvP qu'il n'a pas
réellement (le jeu coûte 29,99€ et n'a que le tag Co-op, pas PvP). Le niveau
1 seul ne peut pas détecter ce genre d'erreur puisque le nom du jeu est bien
réel -- d'où l'extension.

Usage :
    from grounding_check import check_grounding

    report = check_grounding(response_text, search_results)
    if report["unmatched"] or report["fact_issues"]:
        print(format_grounding_warning(report))
"""

import re

from rapidfuzz import fuzz

# Cible le gras markdown UNIQUEMENT en position de titre :
#   - début de ligne de liste numérotée/à puces : "1. **Nom**", "- **Nom**"
#   - première cellule d'une ligne de tableau markdown : "| **Nom** | ..."
# Exclut volontairement le gras utilisé comme emphase dans le corps du texte
# (ex: "un jeu **casual** et **relaxant**"), qui n'est pas un nom de jeu cité
# mais un simple style rédactionnel -- traiter tout gras comme candidat
# produisait des faux positifs sur ces adjectifs.
LIST_TITLE_PATTERN = re.compile(r'^\s*(?:\d+[.)]|[-*])\s*\*\*(.+?)\*\*', re.MULTILINE)
TABLE_TITLE_PATTERN = re.compile(r'^\s*\|\s*\*\*(.+?)\*\*', re.MULTILINE)

MATCH_THRESHOLD = 85.0

# Motifs de prix : "29,99€", "29.99 €", "€29.99"...
PRICE_PATTERN = re.compile(r'(\d+(?:[.,]\d{1,2})?)\s*€|€\s*(\d+(?:[.,]\d{1,2})?)')
FREE_PATTERN = re.compile(r'\b(free[\s-]?to[\s-]?play|free|gratuit)\b', re.IGNORECASE)

# Mots-clés de mécanique surveillés : mot-clé -> sous-chaîne attendue dans les
# vrais tags du jeu (en minuscules) pour que la mention soit considérée fondée.
MECHANIC_KEYWORDS = {
    "pvp": "pvp",
    "co-op": "co-op",
    "coop": "co-op",
    "cooperative": "co-op",
    "multiplayer": "multiplayer",
    "singleplayer": "singleplayer",
    "single-player": "singleplayer",
}


def _extract_cited_titles_with_span(response_text: str) -> list[tuple[str, int, int]]:
    """Comme _extract_cited_names, mais conserve aussi la position (start, end)
    du match pour pouvoir extraire le reste de la ligne (le "texte de claim")."""
    results = []
    for pattern in (LIST_TITLE_PATTERN, TABLE_TITLE_PATTERN):
        for m in pattern.finditer(response_text):
            title = m.group(1).strip()
            if title:
                results.append((title, m.start(), m.end()))
    return results


def _line_containing(text: str, pos: int) -> str:
    """Retourne la ligne complète du texte contenant la position pos."""
    line_start = text.rfind('\n', 0, pos) + 1
    line_end = text.find('\n', pos)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def _check_price_claim(claim_text: str, real_price: float | None) -> str | None:
    """Compare un prix/gratuité mentionné dans claim_text au vrai prix.
    Retourne un message d'anomalie, ou None si cohérent ou rien à vérifier."""
    if real_price is None:
        return None

    says_free = bool(FREE_PATTERN.search(claim_text))
    price_match = PRICE_PATTERN.search(claim_text)

    if says_free and real_price > 0:
        return f"annoncé comme gratuit/free mais coûte en réalité {real_price}€"

    if price_match:
        raw = price_match.group(1) or price_match.group(2)
        claimed_price = float(raw.replace(',', '.'))
        if abs(claimed_price - real_price) > 0.5:  # tolérance pour arrondis
            return f"prix cité {claimed_price}€ mais le vrai prix est {real_price}€"

    return None


def _check_mechanic_claims(claim_text: str, real_tags: list[str]) -> list[str]:
    """Compare les mécaniques mentionnées (PvP, Co-op...) aux vrais tags du jeu.
    Retourne la liste des mécaniques citées mais absentes des vrais tags."""
    real_tags_lower = " ".join(real_tags).lower()
    issues = []
    claim_lower = claim_text.lower()

    for keyword, expected_substring in MECHANIC_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', claim_lower):
            if expected_substring not in real_tags_lower:
                issues.append(f"mécanique '{keyword}' citée mais absente des vrais tags ({real_tags})")

    return issues


def check_grounding(response_text: str, search_results: list[dict]) -> dict:
    """Vérifie la réponse à deux niveaux : existence des jeux cités (noms),
    et véracité des faits cités à côté de chaque nom (prix, mécaniques).

    Retourne un dict avec :
      - matched : liste de (cité, nom_reel, score)
      - unmatched : noms cités sans correspondance -> jeu potentiellement inventé
      - fact_issues : liste de (nom_reel, description_du_probleme) -> jeu réel
        mais détail cité erroné (prix, mécanique...)
      - real_names : les vrais noms disponibles, pour contexte dans le warning
    """
    real_names = [r["name"] for r in search_results]
    results_by_name = {r["name"]: r for r in search_results}

    cited = _extract_cited_titles_with_span(response_text)

    matched = []
    unmatched = []
    fact_issues = []

    for candidate, start, end in cited:
        best_name, best_score = None, 0.0
        for real_name in real_names:
            score = fuzz.ratio(candidate.lower(), real_name.lower())
            if score > best_score:
                best_name, best_score = real_name, score

        if best_score < MATCH_THRESHOLD:
            unmatched.append(candidate)
            continue

        matched.append((candidate, best_name, best_score))

        # Le jeu existe bien -> on vérifie les faits cités près de son nom.
        # On prend la ligne complète (contient souvent prix/tags à côté du nom
        # dans un tableau, ou juste après dans une liste).
        claim_text = _line_containing(response_text, start)
        real_result = results_by_name[best_name]

        price_issue = _check_price_claim(claim_text, real_result.get("price"))
        if price_issue:
            fact_issues.append((best_name, price_issue))

        for mechanic_issue in _check_mechanic_claims(claim_text, real_result.get("tags", [])):
            fact_issues.append((best_name, mechanic_issue))

    return {
        "matched": matched,
        "unmatched": unmatched,
        "fact_issues": fact_issues,
        "real_names": real_names,
    }


def format_grounding_warning(report: dict) -> str | None:
    """Retourne un message d'avertissement lisible s'il y a des noms non
    ancrés ou des faits erronés, sinon None."""
    parts = []

    if report["unmatched"]:
        parts.append(
            f"Noms cités sans correspondance dans les résultats : {report['unmatched']}\n"
            f"Jeux réellement disponibles : {report['real_names']}"
        )

    if report["fact_issues"]:
        issues_str = "\n".join(f"  - {name}: {issue}" for name, issue in report["fact_issues"])
        parts.append(f"Faits cités incohérents avec les vraies données :\n{issues_str}")

    if not parts:
        return None

    return "[AVERTISSEMENT GROUNDING]\n" + "\n".join(parts)