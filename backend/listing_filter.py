"""
Shared listing filtering and autocomplete validation logic.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Avito category words that are safe to select via autocomplete (still used by Playwright)
CATEGORY_KEYWORDS = {
    "комплектующие", "процессоры", "процессор",
    "видеокарты", "видеокарта",
    "оперативная память", "оператив",
    "материнские платы", "материнск",
    "блоки питания", "блок питания",
    "корпуса", "корпус",
    "охлаждение", "кулеры", "кулер",
    "накопители", "накопитель",
    "ssd", "hdd",
    "вентиляторы", "вентилятор",
    "радиаторы", "радиатор",
}

REJECT_KEYWORDS = {
    "пк", "сборк", "компьютер", "системн",
    "ноутбук", "готов", "монитор",
    "связк", "комплект", "набор", "клавиатур",
    "мышк", "принтер",
}

BUNDLE_SYMBOLS = {"+", "/", "\\"}


def is_suggestion_compatible(query: str, suggestion_text: str) -> bool:
    """Check if an autocomplete suggestion is safe to select."""
    if not suggestion_text:
        return False
    q_lower = query.lower()
    s_lower = suggestion_text.lower()
    query_words = set(q_lower.split())
    if any(w in s_lower for w in query_words if len(w) > 1):
        return True
    for cat_word in CATEGORY_KEYWORDS:
        if cat_word in s_lower:
            return True
    return False


def _normalize(text: str) -> str:
    """Lowercase, remove non-alphanumeric except spaces and digits."""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


GENERIC_WORDS = {
    "ram", "cpu", "gpu", "psu", "pc", "box", "oem", "tray", "retail", "new",
    "ssd", "hdd", "nvme", "m2",
}


def _get_core_words(query: str) -> list[str]:
    """Extract specific model words from the end of the query (most specific part)."""
    q = _normalize(query)
    words = [w for w in q.split() if w not in GENERIC_WORDS]
    core = []
    for w in reversed(words):
        if len(core) >= 2:
            break
        has_num = bool(re.search(r'\d', w))
        if (has_num and len(w) >= 2) or len(w) >= 3:
            core.append(w)
    return core[::-1] if core else words[-1:]


def _contains_model(title: str, search_query: str) -> bool:
    q = _normalize(search_query)
    t = _normalize(title)
    q_words = q.split()
    t_words = t.split()
    t_joined = ' '.join(t_words)

    if not q or not t:
        return False

    if q in t:
        return True

    core = _get_core_words(search_query)

    # Strict: all core words must appear in title
    if len(core) >= 2:
        return all(cw in t_joined for cw in core)

    # Single core word
    if len(core) == 1:
        if core[0] in t_joined:
            return True
        # If core is a pure number like "3070", also allow it as standalone word
        if re.match(r'^\d+$', core[0]):
            return core[0] in t_words
        return False

    # No core words: fallback to common words
    common = set(q_words) & set(t_words)
    if len(common) >= 2:
        return True

    q_nums = set(re.findall(r'\d{3,}', q))
    t_nums = set(re.findall(r'\d{3,}', t))
    return bool(q_nums and q_nums & t_nums)

    return False


def _has_reject_keywords(title: str, search_query: str = "") -> bool:
    """Check if title contains keywords indicating a whole PC, bundle, or non-component."""
    t = title.lower()
    q = search_query.lower()
    for kw in REJECT_KEYWORDS:
        if kw in t and kw not in q:
            return True
    for sym in BUNDLE_SYMBOLS:
        if sym in t:
            return True
    return False


def filter_listings(listings: list[tuple[str, float]], component_type: str, search_query: str = "") -> list[float]:
    """
    Filter listings using rule-based checks:
    - Must contain model identifier from search query in title
    - Must not contain PC/build/bundle keywords
    """
    if not listings:
        return []

    if not component_type or component_type == "other":
        return [price for _, price in listings]

    filtered = []
    skipped_reject = 0
    skipped_model = 0

    for title, price in listings:
        if not title:
            continue

        if _has_reject_keywords(title, search_query):
            skipped_reject += 1
            continue

        if not _contains_model(title, search_query):
            skipped_model += 1
            continue

        filtered.append(price)

    logger.info(f"Filtered {len(filtered)}/{len(listings)} for {search_query} "
                f"(reject_kw: {skipped_reject}, no_model: {skipped_model})")
    return filtered
