"""
Selenium-based Avito parser. Opens a new browser per call, navigates directly to search URL.
"""
import logging
import re
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException

from backend.config import AVITO_BASE_URL, MAX_LISTINGS, PROXY_URL
from backend.listing_filter import filter_listings

logger = logging.getLogger(__name__)


def _create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    if PROXY_URL:
        options.add_argument(f"--proxy-server={PROXY_URL}")
    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => false});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
        """
    })
    return driver


def _is_page_blocked(driver) -> bool:
    try:
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        return any(kw in body for kw in
                   ["доступ ограничен", "капча", "captcha", "робот",
                    "подтвердите", "verify", "429",
                    "что нужно сделать"])
    except Exception:
        return True


def _wait_for_block_clear(driver, max_wait: int = 300) -> bool:
    if not _is_page_blocked(driver):
        return True
    logger.warning("Avito blocked! Solve captcha in the Chrome window.")
    logger.warning(f"Waiting up to {max_wait}s for captcha to be solved...")
    import time as _time
    waited = 0
    while waited < max_wait:
        _time.sleep(3)
        waited += 3
        if not _is_page_blocked(driver):
            logger.info("Captcha solved, continuing!")
            return True
        if waited % 30 == 0:
            logger.info(f"Still waiting for captcha... ({waited}s)")
    logger.error("Timed out waiting for captcha")
    return False


def _safe_get(driver, url: str, retries=1):
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            return
        except TimeoutException:
            if attempt < retries:
                logger.warning(f"Page load timeout, retrying ({attempt+1}/{retries})")
            else:
                raise
        except WebDriverException:
            raise


def _extract_all_listings(driver, max_listings: int) -> list[tuple[str, float]]:
    all_listings = []
    items = driver.find_elements(By.CSS_SELECTOR, '[data-marker="item"]')
    for item in items[:max_listings]:
        try:
            title_el = item.find_element(By.CSS_SELECTOR, '[itemprop="name"], [data-marker*="title"], h3')
            title = title_el.text.strip()
        except Exception:
            title = ""
        try:
            meta = item.find_element(By.CSS_SELECTOR, 'meta[itemprop="price"]')
            p = float(meta.get_attribute("content"))
            if 100 <= p <= 10_000_000:
                all_listings.append((title, p))
        except Exception:
            pass
    return all_listings[:max_listings]


def parse_avito_prices_selenium(search_query: str, max_listings: int = MAX_LISTINGS, component_type: str = "") -> list[float]:
    """Open Chrome, navigate directly to Avito search URL, extract and return prices."""
    from backend.parser import BOOTCACHE_SENTINEL

    logger.info(f"Opening Chrome for: {search_query}")
    driver = _create_driver()

    try:
        encoded = quote(search_query)
        _safe_get(driver, f"{AVITO_BASE_URL}/all?q={encoded}")

        if _is_page_blocked(driver):
            if not _wait_for_block_clear(driver):
                return BOOTCACHE_SENTINEL
            _safe_get(driver, f"{AVITO_BASE_URL}/all?q={encoded}")
            if _is_page_blocked(driver):
                return BOOTCACHE_SENTINEL

        all_raw = _extract_all_listings(driver, max_listings)
        logger.info(f"Page 1: {len(all_raw)} raw listings for: {search_query}")

        prices = filter_listings(all_raw, component_type, search_query) if component_type else [p for _, p in all_raw]
        logger.info(f"Total: {len(all_raw)} raw -> {len(prices)} verified for: {search_query}")
        return prices

    except TimeoutException:
        logger.error("Page load timeout — Avito not responding or browser was closed")
        return []
    except WebDriverException as e:
        emsg = str(e).lower()
        if "invalid session" in emsg or "no such window" in emsg or "target window already closed" in emsg:
            logger.error("Browser was closed during parsing")
        else:
            logger.error(f"Selenium error: {e}")
        return []
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def close_browser():
    pass
