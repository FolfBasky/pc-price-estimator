"""
AI-based parser for build descriptions using a local LLM (transformers + CUDA).
Falls back to regex if the model is not ready or fails.

Model: Qwen2.5-1.5B-Instruct (runs on GPU via PyTorch).
"""
import json
import logging
import os
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

_loaded = False
_loaded_error = None
_load_status = "idle"  # idle → loading → ready | error
_model = None
_tokenizer = None
_device = None
_load_lock = threading.Lock()


def _get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(force_reload=False):
    """
    Load the model into GPU memory. Called once at startup.
    Thread-safe, idempotent.
    """
    global _model, _tokenizer, _device, _loaded, _load_status, _loaded_error

    if _loaded and not force_reload:
        return True

    with _load_lock:
        if _loaded and not force_reload:
            return True

        _load_status = "loading"
        _loaded_error = None
        device = _get_device()
        logger.info(f"Loading model {MODEL_NAME} on {device}...")
        logger.info("This will download ~6GB on first run and may take a minute.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                model = model.to("cpu")

            _model = model
            _tokenizer = tokenizer
            _device = device
            _loaded = True
            _load_status = "ready"
            logger.info(f"Model loaded on {device}. VRAM: ~6GB")
            return True
        except Exception as e:
            _load_status = "error"
            _loaded_error = str(e)
            logger.error(f"Failed to load model: {e}")
            return False


_SYSTEM_PROMPT = """Извлеки все комплектующие ПК из текста. Перечисли всё, что упоминается.

Правила:
- Ищи ВСЕ типы: cpu, gpu, ram, motherboard, storage, psu, case, cooler
- Извлекай даже если компонент просто упомянут в описании: "корпус имеет", "система охлаждения", "сжо", "кулеры"
- Включай модель, бренд, объём в search_query
- Если сказано "без [компонента]" — не включай его
- Для case/cooler: если нет модели/бренда, а только описание (прозрачная панель, подсветка, вентиляторы) — пиши просто "игровой корпус" или "универсальный кулер"
- Не выдумывай
- Если компонент не указан явно или нет модели — не добавляй его

Примеры:
"Процессор: AMD Ryzen 5 5600, память 16GB, корпус NZXT, SSD 500GB, СЖО"
→ [{"component_type":"cpu","search_query":"AMD Ryzen 5 5600"},{"component_type":"ram","search_query":"16GB DDR4"},{"component_type":"case","search_query":"корпус NZXT"},{"component_type":"storage","search_query":"500GB SSD"},{"component_type":"cooler","search_query":"СЖО"}]

"Материнская плата: MSI B560, Видеокарта: RTX 3060, корпус имеет стеклянные панели, установлено охлаждение"
→ [{"component_type":"motherboard","search_query":"MSI B560"},{"component_type":"gpu","search_query":"RTX 3060"},{"component_type":"case","search_query":"игровой корпус"},{"component_type":"cooler","search_query":"универсальный кулер"}]

"Процессор: Ryzen 9. Без видеокарты."
→ [{"component_type":"cpu","search_query":"Ryzen 9"}]

"Корпус с прозрачной боковой панелью и подсветкой"
→ [{"component_type":"case","search_query":"игровой корпус"}]

Ответь ТОЛЬКО JSON-массивом."""


def _generate_text(messages: list[dict], max_tokens: int = 512) -> str | None:
    """Generate a response from the AI model. Returns generated text or None."""
    global _model, _tokenizer, _device, _loaded
    if not _loaded or _model is None:
        return None
    try:
        prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )
        return _tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        return None


def parse_with_ai(text: str) -> list[dict] | None:
    """
    Parse a build description using the AI model.
    Returns a list of {component_type, search_query} dicts, or None if AI is unavailable.
    """
    response = _generate_text([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ])
    if response is None:
        return None
    logger.info(f"AI raw response: {response[:200]}")
    return _parse_json_response(response)


def _parse_json_response(text: str) -> list[dict] | None:
    """Extract a JSON array from the model's text response."""
    # Try to find a JSON array in the response
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                valid = []
                for item in items:
                    ct = item.get("component_type", "")
                    sq = item.get("search_query", "")
                    if ct and sq and ct in _VALID_TYPES:
                        valid.append({"component_type": ct, "search_query": sq})
                if valid:
                    return valid
        except (json.JSONDecodeError, Exception):
            pass
    return None


_VALID_TYPES = {"cpu", "gpu", "ram", "motherboard", "storage", "psu", "case", "cooler", "other"}

# Type keyword mapping for regex fallback
_TYPE_KEYWORDS = {
    "cpu": ["процессор", "cpu", "ryzen", "intel", "core", "i3", "i5", "i7", "i9", "amd"],
    "gpu": ["видеокарта", "gpu", "rtx", "gtx", "radeon", "rx ", "geforce", " quadro"],
    "ram": ["память", "ram", "ddr", "оператив", "озу", "gb "],
    "motherboard": ["материнск", "motherboard", "плата", "b560", "b660", "b760", "z690", "z790", "x570", "b550", "h610", "a520"],
    "storage": ["накопитель", "ssd", "hdd", "nvme", "m.2", "твердотель", "жёстк", "жестк", "диск"],
    "psu": ["блок питания", "psu", "бп", " watt", "w ", "corsair ", "cougar", "chieftec"],
    "case": ["корпус", "case"],
    "cooler": ["охлаждение", "cooler", "кулер", "радиатор", "вентилятор", "воздуш"],
    "other": [],
}


def _split_parts(text: str) -> list[str]:
    """Split text into candidate part strings."""
    # First, try splitting by common separators
    parts = re.split(r'[,;:\n/•|]+', text)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]

    # No clear separators — split on type keywords
    all_kws = sorted(set(kw for kws in _TYPE_KEYWORDS.values() for kw in kws), key=len, reverse=True)
    pattern = '|'.join(re.escape(kw) for kw in all_kws)
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if len(matches) >= 2:
        parts = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip().strip(',;:')
            if chunk:
                parts.append(chunk)
        return parts

    return [text]


def parse_with_regex(text: str) -> list[dict]:
    """
    Fallback regex parser for build descriptions.
    """
    parts = _split_parts(text)
    result = []
    seen_queries = set()

    for part in parts:
        if len(part) < 3:
            continue

        lower = part.lower()
        best_type = None
        best_score = 0

        for ctype, keywords in _TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > best_score:
                best_score = score
                best_type = ctype

        if best_type:
            # Keep the whole part as the query (just trim punctuation)
            query = part.strip().strip(',;:- ')
            query = re.sub(r'\s+', ' ', query).strip()
            if query and len(query) > 1 and query.lower() not in seen_queries:
                seen_queries.add(query.lower())
                result.append({"component_type": best_type, "search_query": query})
            elif query and len(query) > 1:
                # Duplicate — try appending to last with same type
                pass

    return result


# Latin homoglyphs that look like Cyrillic but are different Unicode characters
_HOMOGLYPH_MAP = {
    'a': 'а', 'A': 'А',  # Latin a → Cyrillic а
    'e': 'е', 'E': 'Е',  # Latin e → Cyrillic е
    'o': 'о', 'O': 'О',  # Latin o → Cyrillic о
    'p': 'р', 'P': 'Р',  # Latin p → Cyrillic р
    'c': 'с', 'C': 'С',  # Latin c → Cyrillic с
    'x': 'х', 'X': 'Х',  # Latin x → Cyrillic х
    'y': 'у', 'Y': 'У',  # Latin y → Cyrillic у
    'i': 'і', 'I': 'І',  # Latin i → Cyrillic і (rare, but possible)
}


def _normalize_homoglyphs(text: str) -> str:
    """
    Replace Latin homoglyphs with Cyrillic equivalents in mostly-Cyrillic text.
    Only replaces in runs of Cyrillic text to avoid mangling English queries.
    """
    # Count Cyrillic vs Latin chars
    cyrillic_count = sum(1 for ch in text if '\u0400' <= ch <= '\u04FF')
    latin_count = sum(1 for ch in text if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z')

    # If mostly English, don't touch
    if latin_count > cyrillic_count:
        return text

    result = []
    for ch in text:
        if ch in _HOMOGLYPH_MAP:
            result.append(_HOMOGLYPH_MAP[ch])
        else:
            result.append(ch)
    return ''.join(result)


# Mapping from exclusion keywords to component types
_EXCLUDE_KEYWORDS = {
    "gpu": ["видеокарт", "видео", "gpu", "графическ"],
    "ram": ["оператив", "озу", "ram", "память"],
    "storage": ["ssd", "hdd", "накопител", "диск", "nvme"],
    "cpu": ["процессор", "cpu"],
    "motherboard": ["материнск", "плат"],
    "psu": ["блок", "питани", "psu", "бп"],
    "case": ["корпус"],
    "cooler": ["охлажден", "кулер", "сжо"],
}


def _filter_excluded(items: list[dict], original_text: str) -> list[dict]:
    """
    Post-processing: if the text says "без [component]", remove it from results.
    Checks for patterns like "без видеокарты", "без оперативной памяти",
    "без видеокарты, оперативной памяти и SSD".
    """
    lower = original_text.lower()
    # Find "без ..." clauses — capture until end of sentence or comma-separated list
    bez_match = re.search(r'\bбез\s+(.+?)(?:[.。\n]|$)', lower, re.DOTALL)
    if not bez_match:
        return items

    bez_text = bez_match.group(1).strip().rstrip(',;')
    # Expand comma/и separated list
    bez_parts = re.split(r'[,;]|\sи\s', bez_text)

    excluded_types = set()
    for part in bez_parts:
        part = part.strip()
        if not part:
            continue
        for ctype, keywords in _EXCLUDE_KEYWORDS.items():
            for kw in keywords:
                if kw in part:
                    excluded_types.add(ctype)
                    break

    if not excluded_types:
        return items

    return [item for item in items if item["component_type"] not in excluded_types]


def _deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicates by component_type, keep first. Filter empty queries."""
    seen = set()
    result = []
    for item in items:
        ct = item.get("component_type", "")
        sq = item.get("search_query", "").strip()
        if not ct or not sq:
            continue
        if ct not in seen:
            seen.add(ct)
            result.append({"component_type": ct, "search_query": sq})
    return result


def parse_description(text: str) -> list[dict]:
    """
    Parse build description using AI model only.
    Returns deduplicated list of {component_type, search_query}.
    """
    normalized = _normalize_homoglyphs(text)
    ai_result = parse_with_ai(normalized)
    if ai_result:
        deduped = _deduplicate(ai_result)
        logger.info(f"AI parsed {len(ai_result)} components → {len(deduped)} after dedup")
        return deduped

    return []


def get_status() -> dict:
    """Return AI model loading status."""
    return {
        "status": _load_status,
        "error": _loaded_error,
        "ready": _loaded and _model is not None,
    }


def is_ready() -> bool:
    """Check if the AI model is loaded and ready."""
    return _loaded and _model is not None


_VERIFY_SYSTEM_PROMPT = """Твоя задача — определить, является ли объявление продажей отдельного компонента по запросу.
Ответь "да", если это продажа отдельного компонента (не в сборке и не с другими компонентами в лоте).
Ответь "нет", если это: готовый ПК, ноутбук, связка нескольких компонентов, или другой товар.
Ответь только "да" или "нет", без пояснений.

Примеры:
Запрос: i7-12700K
Заголовок: Процессор Intel Core i7-12700K OEM
→ да
Заголовок: Игровой ПК на i7-12700K + RTX 4070
→ нет
Заголовок: I7 12700k + мать Z690
→ нет
Заголовок: Intel Core i7-12700K
→ да"""


def verify_listing(title: str, component_type: str, search_query: str) -> bool:
    """
    AI-based verification: check if a listing title actually matches the expected component.
    Returns True if it's a match, False if not (defaults to True if AI unavailable).
    """
    response = _generate_text([
        {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Запрос: {search_query}\nЗаголовок: {title}"},
    ], max_tokens=10)

    if response is None:
        return True

    decision = response.strip().lower()
    if "да" in decision:
        return True
    return False
