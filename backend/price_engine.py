import asyncio
import logging
import random
from datetime import datetime, timedelta
from statistics import median

from backend.config import CACHE_TTL_DAYS, PARSER_DELAY_MIN, PARSER_DELAY_MAX, MAX_LISTINGS
from backend.database import SessionLocal, PriceCache, PriceHistory, Build, BuildItem, ComponentType
from backend.parser import parse_avito_prices as parse_playwright, BOOTCACHE_SENTINEL

try:
    from backend.parser_cffi import parse_avito_prices_cffi as parse_cffi
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False

_SELENIUM_AVAILABLE = None  # lazy check

# Generic fallback queries and min prices per component slug
_GENERIC_QUERIES = {
    "case": "игровой корпус",
    "cooler": "универсальный кулер",
    "psu": "блок питания 600w",
    "storage": "ssd 500gb",
    "ram": "оперативная память 16gb",
    "cpu": "процессор amd ryzen",
    "gpu": "видеокарта",
    "motherboard": "материнская плата",
    "other": "комплектующие",
}

_MIN_PRICES = {
    "cpu": 2000,
    "gpu": 3000,
    "ram": 1000,
    "motherboard": 1500,
    "storage": 1000,
    "psu": 1000,
    "case": 2000,
    "cooler": 500,
    "other": 500,
}

logger = logging.getLogger(__name__)


def calc_stats(raw_prices: list[float]) -> dict:
    if not raw_prices:
        return {"avg": None, "median": None, "min": None, "max": None, "listings_raw": 0, "listings": 0}

    sane = [p for p in raw_prices if 500 <= p <= 5_000_000]
    if len(sane) < 3:
        sane = sorted(raw_prices)

    sorted_prices = sorted(sane)
    n = len(sorted_prices)

    # Cluster-based filtering: find the largest price gap (>1.5x),
    # take the cluster before it (cheapest items, likely correct component)
    # If the cheap cluster is too small (<3 items), take the next one
    best_idx = n
    max_ratio = 0
    for i in range(1, n):
        if sorted_prices[i - 1] > 0:
            r = sorted_prices[i] / sorted_prices[i - 1]
            if r > max_ratio and r > 1.5:
                max_ratio = r
                best_idx = i

    if best_idx < n:
        cluster = sorted_prices[:best_idx]
        if len(cluster) < 3 and best_idx + 3 <= n:
            cluster = sorted_prices[best_idx:]
    else:
        cluster = sorted_prices

    return {
        "avg": round(sum(cluster) / len(cluster), 2),
        "median": round(median(cluster), 2),
        "min": round(min(cluster), 2),
        "max": round(max(cluster), 2),
        "listings_raw": len(raw_prices),
        "listings": len(cluster),
    }


def get_cached_price(db, comp_type_id: int, query: str):
    cutoff = datetime.utcnow() - timedelta(days=CACHE_TTL_DAYS)
    return (
        db.query(PriceCache)
        .filter(
            PriceCache.component_type_id == comp_type_id,
            PriceCache.search_query == query,
            PriceCache.parsed_at >= cutoff,
        )
        .order_by(PriceCache.parsed_at.desc())
        .first()
    )


def save_cache(db, comp_type_id: int, query: str, stats: dict):
    entry = PriceCache(
        component_type_id=comp_type_id,
        search_query=query,
        avg_price=stats["avg"],
        median_price=stats["median"],
        min_price=stats["min"],
        max_price=stats["max"],
        listings_count=stats["listings"],
        listings_raw=stats["listings_raw"],
        parsed_at=datetime.utcnow(),
    )
    db.add(entry)

    hist = PriceHistory(
        component_type_id=comp_type_id,
        search_query=query,
        avg_price=stats["avg"],
        median_price=stats["median"],
        min_price=stats["min"],
        max_price=stats["max"],
        parsed_at=datetime.utcnow(),
    )
    db.add(hist)
    db.commit()
    db.refresh(entry)
    return entry


async def _process_item(db, item, comp_type) -> tuple[dict | None, bool, bool]:
    """Parse a single item and save its price. Returns (stats, avito_blocked, query_changed)."""
    from backend.parser import BOOTCACHE_SENTINEL

    item.status = "running"
    db.commit()

    cached = get_cached_price(db, item.component_type_id, item.search_query)
    if cached:
        stats = {
            "avg": float(cached.avg_price) if cached.avg_price else None,
            "median": float(cached.median_price) if cached.median_price else None,
            "min": float(cached.min_price) if cached.min_price else None,
            "max": float(cached.max_price) if cached.max_price else None,
            "listings": cached.listings_count,
        }
        item.price_cache_id = cached.id
        item.status = "done"
        db.commit()
        return stats, False, False

    result = []
    avito_blocked = False

    async def run_parsers(query: str) -> list | object:
        nonlocal avito_blocked
        res: list | object = []

        try:
            from backend.parser_selenium import parse_avito_prices_selenium
            res = await asyncio.wait_for(
                asyncio.to_thread(parse_avito_prices_selenium, query, MAX_LISTINGS, comp_type),
                timeout=120,
            )
        except ImportError:
            pass
        except asyncio.TimeoutError:
            logger.error(f"Selenium timed out for: {query}")
        except Exception as e:
            logger.error(f"Selenium parse failed: {e}")

        if not res and HAS_CFFI:
            try:
                res = await parse_cffi(query, MAX_LISTINGS, comp_type)
            except Exception as e:
                logger.error(f"CFFI parse failed for {query}: {e}")

        if not res or (isinstance(res, list) and len(res) == 0):
            try:
                res = await parse_playwright(query, MAX_LISTINGS, comp_type)
            except Exception as e:
                logger.error(f"Playwright parse failed for {query}: {e}")
                res = res or []

        if res is BOOTCACHE_SENTINEL:
            avito_blocked = True
        return res

    result = await run_parsers(item.search_query)

    if result is BOOTCACHE_SENTINEL:
        stats = {"avg": None, "median": None, "min": None, "max": None, "listings_raw": 0, "listings": -1}
        item.status = "error"

    elif result:
        stats = calc_stats(result)
        cache_entry = save_cache(db, item.component_type_id, item.search_query, stats)
        item.price_cache_id = cache_entry.id
        item.status = "done"

    else:
        generic_q = _GENERIC_QUERIES.get(comp_type)
        if generic_q and generic_q.lower() != item.search_query.lower():
            logger.info(f"No results for '{item.search_query}', retrying generic '{generic_q}'")
            retry_result = await run_parsers(generic_q)
            if retry_result and retry_result is not BOOTCACHE_SENTINEL:
                stats = calc_stats(retry_result)
                cache_entry = save_cache(db, item.component_type_id, generic_q, stats)
                item.price_cache_id = cache_entry.id
                item.search_query = generic_q
                item.status = "done"
                db.commit()
                return stats, avito_blocked, True

        min_price = _MIN_PRICES.get(comp_type)
        if min_price:
            logger.info(f"Using min price {min_price}₽ for {comp_type}")
            stats = {"avg": float(min_price), "median": float(min_price), "min": float(min_price), "max": float(min_price), "listings_raw": 1, "listings": 1}
            cache_entry = save_cache(db, item.component_type_id, item.search_query, stats)
            item.price_cache_id = cache_entry.id
            item.status = "done"
        else:
            stats = {"avg": None, "median": None, "min": None, "max": None, "listings_raw": 0, "listings": 0}
            item.status = "error"

    db.commit()
    return stats, avito_blocked, False


async def process_build(build_id: int):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            return

        build.status = "running"
        items = db.query(BuildItem).filter(
            BuildItem.build_id == build_id,
            BuildItem.is_hidden == 0,
        ).order_by(BuildItem.sort_order, BuildItem.id).all()
        db.commit()

        avito_blocked = False
        for i, item in enumerate(items):
            ct = db.query(ComponentType).filter(ComponentType.id == item.component_type_id).first()
            comp_type = ct.slug if ct else ""

            _, item_blocked, _ = await _process_item(db, item, comp_type)
            if item_blocked:
                avito_blocked = True

            build.progress = int((i + 1) / len(items) * 100)
            db.commit()

            if i < len(items) - 1:
                await asyncio.sleep(random.uniform(PARSER_DELAY_MIN, PARSER_DELAY_MAX))

        total = 0.0
        all_done = True
        for item in items:
            if item.price_cache_id:
                pc = db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).first()
                if pc and pc.median_price:
                    total += pc.median_price
            if item.status != "done":
                all_done = False

        build.total_price = round(total, 2) if total > 0 else None
        build.status = "blocked" if avito_blocked else ("done" if all_done else "partial")
        build.progress = 100
        db.commit()

    except Exception as e:
        logger.error(f"Error processing build {build_id}: {e}")
        build = db.query(Build).filter(Build.id == build_id).first()
        if build:
            build.status = "error"
            db.commit()
    finally:
        db.close()


async def parse_single_item(build_id: int, item_id: int):
    """Parse a single item without re-processing the entire build."""
    db = SessionLocal()
    try:
        item = db.query(BuildItem).filter(
            BuildItem.id == item_id,
            BuildItem.build_id == build_id,
        ).first()
        if not item:
            return

        ct = db.query(ComponentType).filter(ComponentType.id == item.component_type_id).first()
        comp_type = ct.slug if ct else ""

        await _process_item(db, item, comp_type)

        # Recalculate total
        total = 0.0
        items = db.query(BuildItem).filter(
            BuildItem.build_id == build_id,
            BuildItem.is_hidden == 0,
        ).all()
        for it in items:
            if it.price_cache_id:
                pc = db.query(PriceCache).filter(PriceCache.id == it.price_cache_id).first()
                if pc and pc.median_price:
                    total += pc.median_price

        build = db.query(Build).filter(Build.id == build_id).first()
        if build:
            build.total_price = round(total, 2) if total > 0 else None
            if all(it.status == "done" for it in items):
                build.status = "done"
            db.commit()

    except Exception as e:
        logger.error(f"Error parsing single item {item_id}: {e}")
    finally:
        db.close()
