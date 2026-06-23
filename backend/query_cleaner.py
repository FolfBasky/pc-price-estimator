import re

# Only truly non-informative words for Avito search.
# Component names (корпус, процессор, блок, etc.) are kept — they're essential for search.
STOP_WORDS = {
    "б/у", "бу", "б у",
    "для", "на", "в", "из", "с", "со", "под", "по", "за", "от", "до",
    "и", "а", "но", "или",
    "этот", "эта", "это", "эти",
    "продаю", "продам", "продажа",
    "недорого", "дешево", "дешевый",
    "срочно", "есть",
    "штука", "штуки", "шт",
}


def simplify_query(query: str) -> str:
    """Remove stop words and trim whitespace from a search query."""
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r"\(.*?\)", "", query)
    words = query.split()
    result = [w for w in words if w.lower() not in STOP_WORDS]
    cleaned = " ".join(result)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return query
    return cleaned
