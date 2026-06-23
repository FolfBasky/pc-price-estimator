"""
Playwright-based Avito parser with autocomplete category selection.

Types the search query into Avito's search box to trigger autocomplete,
then clicks the first suggestion to narrow results to the right category.
Falls back to direct search if autocomplete is unavailable.
"""
import asyncio
import logging
import random
import re
from urllib.parse import quote

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from backend.config import AVITO_BASE_URL, PARSER_DELAY_MIN, PARSER_DELAY_MAX, MAX_LISTINGS, HEADLESS, PROXY_URL
from backend.listing_filter import is_suggestion_compatible, filter_listings

logger = logging.getLogger(__name__)

# Sentinels
BOOTCACHE_SENTINEL = object()


async def _find_search_input(page):
    """Find the search input using multiple possible selectors."""
    selectors = [
        'input[data-marker*="search"]',
        'input[data-marker*="suggest"]',
        'input[type="text"][placeholder*="поиск"]',
        'input[type="text"][placeholder*="Поиск"]',
        'input[name="q"]',
    ]
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                visible = await el.is_visible()
                if visible:
                    return el
        except Exception:
            pass
    return None


async def _read_first_suggestion_text(page) -> str | None:
    """Read text of the first autocomplete suggestion."""
    selectors = [
        '[data-marker*="suggest"] li:first-child',
        '[data-marker*="suggest"] div:first-child',
        'ul[data-marker*="suggest"] li:first-child',
        '[class*="suggest"] li:first-child',
    ]
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                text = (await el.inner_text()).strip()
                if text:
                    return text
        except Exception:
            pass
    return None


async def _select_suggestion_if_compatible(page, search_input, query: str) -> bool:
    """Check the first suggestion; if compatible, select it via Arrow Down + Enter."""
    suggestion_text = await _read_first_suggestion_text(page)
    if suggestion_text:
        compat = is_suggestion_compatible(query, suggestion_text)
        logger.info(f"Suggestion: {suggestion_text!r} — {'compatible' if compat else 'incompatible'}")
        if not compat:
            return False

    current_url = page.url
    try:
        await page.keyboard.press("ArrowDown")
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(3000)
        return page.url != current_url
    except Exception as e:
        logger.warning(f"Arrow Down + Enter failed: {e}")
        return False


async def _search_with_autocomplete(page, query: str) -> bool:
    """Type query, trigger autocomplete, select first suggestion."""
    search_input = await _find_search_input(page)
    if not search_input:
        return False

    try:
        await search_input.fill(query)
        await page.wait_for_timeout(1500)

        selected = await _select_suggestion_if_compatible(page, search_input, query)
        if selected:
            logger.info("Compatible suggestion selected via keyboard")
        else:
            logger.info("Suggestion rejected or unavailable, pressing Enter for direct search")
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(3000)
        return True
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return False


async def _extract_price(item) -> float | None:
    try:
        meta = await item.query_selector('meta[itemprop="price"]')
        if meta:
            content = await meta.get_attribute("content")
            if content:
                val = _parse_price_str(content)
                if val is not None:
                    return val
    except Exception:
        pass
    try:
        price_el = await item.query_selector('[data-marker*="price"]')
        if price_el:
            text = await price_el.inner_text()
            val = _parse_price_str(text)
            if val is not None:
                return val
    except Exception:
        pass
    try:
        text_content = await item.inner_text()
        for pat in [r'(\d[\d\s]{1,6}\d)\s*[₽pр]', r'(\d[\d\s]{1,6}\d)\s*руб', r'(\d[\d\s]{1,6}\d)\s*р\.']:
            m = re.search(pat, text_content)
            if m:
                val = _parse_price_str(m.group(1))
                if val is not None:
                    return val
    except Exception:
        pass
    return None


def _parse_price_str(text: str) -> float | None:
    if not text:
        return None
    text = re.sub(r'[^\d\s,.]', '', text.strip())
    text = text.replace('\u00a0', ' ').strip()
    text = text.replace(' ', '')
    text = text.replace(',', '.')
    try:
        val = float(text)
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


async def parse_avito_prices(search_query: str, max_listings: int = MAX_LISTINGS, component_type: str = "") -> list[float]:
    """Parse prices using Playwright with autocomplete category selection."""
    prices = []
    async with async_playwright() as p:
        launch_kwargs = dict(
            headless=HEADLESS,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        if PROXY_URL:
            launch_kwargs["proxy"] = {"server": PROXY_URL}
        browser = await p.chromium.launch(**launch_kwargs)

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="ru-RU",
            timezone_id="Europe/Moscow",
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => false})"
        )

        page = await context.new_page()
        await page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,eot,css,ico}",
                         lambda route: route.abort())
        await asyncio.sleep(random.uniform(1, 3))

        try:
            # Go to Avito all-region page for autocomplete
            await page.goto(f"{AVITO_BASE_URL}/all", wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)

            body_text = await page.inner_text("body").lower()
            if "доступ ограничен" in body_text.lower():
                logger.warning(f"Avito blocked access (IP ban)")
                return BOOTCACHE_SENTINEL
            if "captcha" in body_text.lower() or "капча" in body_text.lower():
                logger.warning(f"Avito CAPTCHA detected")
                return BOOTCACHE_SENTINEL

            # Search using autocomplete
            searched = await _search_with_autocomplete(page, search_query)

            if not searched:
                # Fall back to direct URL
                encoded_query = quote(search_query)
                await page.goto(f"{AVITO_BASE_URL}/all?q={encoded_query}", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(3)

            # Check for block again after search
            body_text = await page.inner_text("body")
            if "доступ ограничен" in body_text.lower():
                return BOOTCACHE_SENTINEL
            if "captcha" in body_text.lower() or "капча" in body_text.lower():
                return BOOTCACHE_SENTINEL

            # Wait for items
            try:
                await page.wait_for_selector('div[data-marker="item"]', timeout=10000)
            except PlaywrightTimeout:
                try:
                    await page.wait_for_selector('[itemtype*="Product"]', timeout=5000)
                except PlaywrightTimeout:
                    pass

            items = await page.query_selector_all('div[data-marker="item"]')
            if not items:
                items = await page.query_selector_all('[itemtype*="Product"]')
            if not items:
                return []

            listings = []
            for item in items[:max_listings]:
                try:
                    title_el = await item.query_selector('[itemprop="name"], [data-marker*="title"], h3')
                    title = (await title_el.inner_text()).strip() if title_el else ""
                except Exception:
                    title = ""
                price = await _extract_price(item)
                if price is not None and 100 <= price <= 10_000_000:
                    listings.append((title, price))

            prices = filter_listings(listings, component_type, search_query) if component_type else [p for _, p in listings]
            logger.info(f"Found {len(listings)} listings → {len(prices)} relevant")

        except PlaywrightTimeout:
            logger.error(f"Timeout loading Avito for: {search_query}")
        except Exception as e:
            logger.error(f"Parse error for '{search_query}': {e}")
        finally:
            await browser.close()

    return prices
