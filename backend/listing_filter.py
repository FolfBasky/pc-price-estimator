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


_GENERIC_CAPACITIES = {
    "8gb", "16gb", "32gb", "64gb", "128gb", "256gb", "512gb",
    "1tb", "2tb", "4tb", "500gb", "120gb", "240gb", "480gb",
}

_GENERIC_WORDS = {
    "ram", "cpu", "gpu", "psu", "pc", "box", "oem", "tray", "retail", "new",
    "ssd", "hdd", "nvme", "m2",
}

_SPECIFIC_BRANDS = {
    "rtx", "gtx", "rx", "radeon", "ryzen", "core", "xeon", "pentium", "celeron",
    "intel", "amd", "nvidia", "geforce", "quadro",
}


def _is_specific_query(query: str) -> bool:
    """Query targets a specific product model (not generic capacity/spec)."""
    q = _normalize(query)
    words = q.split()
    # Mixed letter-digit word like 5600x, x470, 600w
    if re.search(r'(?:\d+[a-zа-яё]+|[a-zа-яё]+\d+)', q):
        return True
    # A ≥4 digit number that isn't a common capacity
    for w in words:
        if re.match(r'^\d{4,}$', w) and w not in _GENERIC_CAPACITIES:
            return True
    # A short number (3+ digits) next to a known brand/series word
    nums = re.findall(r'\d{3,}', q)
    brands = [w for w in words if w in _SPECIFIC_BRANDS]
    if nums and brands:
        return True
    # A distinctive word (≥5 chars, not generic) with a number in the query
    has_distinctive = any(len(w) >= 5 and w not in _GENERIC_WORDS
                          for w in words)
    has_any_number = bool(re.search(r'\d', q))
    if has_distinctive and has_any_number:
        return True
    return False


def _get_identifiers(query: str) -> list[str]:
    """Extract distinctive identifying words from a specific product query."""
    q = _normalize(query)
    words = q.split()
    idents = []
    for w in reversed(words):
        if w in _GENERIC_WORDS or w in _GENERIC_CAPACITIES:
            continue
        if len(w) == 1:
            continue
        if len(w) >= 3 or re.search(r'\d', w):
            idents.append(w)
            if len(idents) >= 2:
                break
    return idents[::-1] if idents else words[-1:]


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

    if _is_specific_query(search_query):
        idents = _get_identifiers(search_query)
        if idents:
            return all(i in t_joined for i in idents)

    # Loose fallback for generic queries
    common = set(q_words) & set(t_words)
    if len(common) >= 2:
        return True
    for w in q_words:
        if re.search(r'\d', w) and w in t_words:
            return True
    q_nums = set(re.findall(r'\d+', q))
    t_nums = set(re.findall(r'\d+', t))
    if q_nums and q_nums & t_nums:
        return True
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
