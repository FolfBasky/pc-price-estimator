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

# Try these models in order (biggest first); fall back to smaller on OOM
_MODEL_SIZES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

_loaded = False
_loaded_error = None
_load_status = "idle"  # idle → loading → ready | error
_model = None
_tokenizer = None
_device = None
_load_lock = threading.Lock()
_current_model_name = ""


def _get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _try_load(model_name: str, device: str) -> tuple:
    """Try to load a specific model. Returns (tokenizer, model) or raises."""
    logger.info(f"Trying {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    return tokenizer, model


def load_model(force_reload=False):
    """
    Load a Qwen model on GPU. Tries 3B → 1.5B → 0.5B if VRAM is insufficient.
    Falls back to CPU if none fit on GPU.
    Thread-safe, idempotent.
    """
    global _model, _tokenizer, _device, _loaded, _load_status, _loaded_error, _current_model_name

    if _loaded and not force_reload:
        return True

    with _load_lock:
        if _loaded and not force_reload:
            return True

        _load_status = "loading"
        _loaded_error = None
        _loaded = False
        device = _get_device()

        models_to_try = list(_MODEL_SIZES)
        last_err = None

        for model_name in models_to_try:
            try:
                _tokenizer, _model = _try_load(model_name, device)
                _device = device
                _loaded = True
                _load_status = "ready"
                _current_model_name = model_name
                vr = "~6GB" if "3B" in model_name else ("~3GB" if "1.5B" in model_name else "~1GB")
                logger.info(f"Loaded {model_name} on {device}. VRAM: {vr}")
                return True
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM loading {model_name} on GPU: {e}")
                last_err = e
                if device == "cuda":
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Cleared CUDA cache, trying smaller model...")
                    continue
                break
            except Exception as e:
                emsg = str(e).lower()
                if "cuda out of memory" in emsg or "out of memory" in emsg:
                    logger.warning(f"OOM loading {model_name} on GPU: {e}")
                    last_err = e
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    if device == "cuda":
                        logger.info("Cleared CUDA cache, trying smaller model...")
                        continue
                last_err = e
                logger.warning(f"Failed to load {model_name}: {e}")
                continue

        # If GPU fails for all sizes, try CPU with the smallest model
        if device == "cuda" and not _loaded:
            logger.warning("All GPU attempts failed, falling back to CPU...")
            device = "cpu"
            smallest = models_to_try[-1]
            try:
                _tokenizer, _model = _try_load(smallest, device)
                _device = device
                _loaded = True
                _load_status = "ready"
                _current_model_name = smallest
                logger.info(f"Loaded {smallest} on CPU (fallback)")
                return True
            except Exception as e:
                last_err = e

        _load_status = "error"
        _loaded_error = str(last_err)
        logger.error(f"Failed to load any model. Last error: {last_err}")
        return False


_SYSTEM_PROMPT = """Извлеки все комплектующие ПК из текста. Перечисли всё, что упоминается.

Правила:
- Ищи ВСЕ типы: cpu, gpu, ram, motherboard, storage, psu, case, cooler
- Извлекай даже если компонент просто упомянут в описании: "корпус имеет", "система охлаждения", "сжо", "кулеры"
- Если сказано "без [компонента]" — не включай его
- Для case/cooler: если нет модели/бренда, а только описание (прозрачная панель, подсветка, вентиляторы) — пиши просто "игровой корпус" или "универсальный кулер"
- Не выдумывай
- Если компонент не указан явно или нет модели — не добавляй его

ВАЖНО: search_query должен быть КОРОТКИМ и подходить для поиска на Avito:
- Для GPU: пиши чипсет с префиксом (RTX 3050, GTX 1660 Super, RX 6600) — без бренда и лишних деталей
- Для CPU: пиши модель (Ryzen 7 5700G, i5-10600)
- Для RAM: пиши "XGB DDRX" (32GB DDR4 → "32GB DDR4")
- Для storage: пиши ёмкость и тип (512GB SSD, 1TB NVMe)
- Для PSU: пиши мощность (600W, 750W)
- Для материнской платы: пиши модель серии (B550M, Z490, X570)

Примеры:
"Процессор: AMD Ryzen 5 5600, память 16GB, корпус NZXT, SSD 500GB, СЖО"
→ [{"component_type":"cpu","search_query":"Ryzen 5 5600"},{"component_type":"ram","search_query":"16GB DDR4"},{"component_type":"case","search_query":"игровой корпус"},{"component_type":"storage","search_query":"500GB SSD"},{"component_type":"cooler","search_query":"СЖО"}]

"Материнская плата: MSI B560, Видеокарта: KFA2 GeForce RTX 3050 CORE 8GB, корпус имеет стеклянные панели"
→ [{"component_type":"motherboard","search_query":"B560"},{"component_type":"gpu","search_query":"RTX 3050"},{"component_type":"case","search_query":"игровой корпус"}]

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


def _simplify_query(query: str, comp_type: str) -> str:
    """Simplify search query for better Avito results — strip brand, keep core model."""
    q = query.strip()
    if comp_type == "gpu":
        m = re.search(r'\b(RTX|GTX|RX|Radeon|Quadro)\s+\w+', q, re.IGNORECASE)
        if m:
            return m.group(0)
        m = re.search(r'(\d{4})\s*(Ti|Super|XT)?', q, re.IGNORECASE)
        if m:
            return m.group(0)
    elif comp_type == "cpu":
        m = re.search(r'(Ryzen\s+\d\s+\d{4}[A-Z]*|Core\s+i\d[-\s]\d{4,5}[A-Z]*|i\d[-\s]\d{4,5}[A-Z]*|Pentium|Celeron|Threadripper)', q, re.IGNORECASE)
        if m:
            return m.group(0)
    elif comp_type == "ram":
        m = re.search(r'(\d+\s*GB).*?(DDR\d)', q, re.IGNORECASE)
        if m:
            return m.group(1) + ' ' + m.group(2)
        m = re.search(r'(\d+\s*GB)', q, re.IGNORECASE)
        if m:
            return m.group(1)
    elif comp_type == "motherboard":
        m = re.search(r'(B\d{3,4}[A-Z]*|H\d{3,4}[A-Z]*|Z\d{3,4}[A-Z]*|X\d{3,4}[A-Z]*|A\d{3,4}[A-Z]*)', q)
        if m:
            return m.group(0)
    elif comp_type == "storage":
        m = re.search(r'(?:(SSD|HDD|NVMe)\s*)?(\d+\s*(?:GB|TB))(?:\s*(SSD|HDD|NVMe))?', q, re.IGNORECASE)
        if m:
            result = m.group(2).strip()
            prefix = m.group(1)
            suffix = m.group(3)
            if prefix:
                result = prefix + ' ' + result
            elif suffix:
                result = suffix + ' ' + result
            return result
    elif comp_type == "psu":
        m = re.search(r'(\d+)\s*W', q, re.IGNORECASE)
        if m:
            return m.group(0)
    return q


def _simplify_queries(items: list[dict]) -> list[dict]:
    """Apply _simplify_query to all items."""
    return [{"component_type": item["component_type"], "search_query": _simplify_query(item["search_query"], item["component_type"])} for item in items]


def parse_description(text: str) -> list[dict]:
    """
    Parse build description using AI model only.
    Returns deduplicated list of {component_type, search_query}.
    """
    normalized = _normalize_homoglyphs(text)
    ai_result = parse_with_ai(normalized)
    if ai_result:
        deduped = _deduplicate(ai_result)
        simplified = _simplify_queries(deduped)
        logger.info(f"AI parsed {len(ai_result)} components → {len(simplified)} after dedup+simplify")
        return simplified

    return []


def get_status() -> dict:
    """Return AI model loading status."""
    return {
        "status": _load_status,
        "error": _loaded_error,
        "ready": _loaded and _model is not None,
        "model": _current_model_name,
        "device": _device,
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
