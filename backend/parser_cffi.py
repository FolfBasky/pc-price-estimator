"""
Alternative Avito parser using curl_cffi (http-level impersonation).
Faster than Playwright, no browser needed.
Falls back to DoH DNS resolution when system DNS is hijacked.
"""
import json
import logging
import random
import re
import socket
import time
from urllib.parse import quote

from curl_cffi import requests

from backend.config import PARSER_DELAY_MIN, PARSER_DELAY_MAX, MAX_LISTINGS, PROXY_URL, AVITO_BASE_URL
from backend.listing_filter import filter_listings

logger = logging.getLogger(__name__)

SUSPICIOUS_IPS = ("198.18.", "10.", "172.16.", "192.168.", "127.", "0.")
AVITO_REAL_IPS = []  # cached real IPs


def _resolve_via_doh(domain: str) -> str | None:
    """Resolve domain via Cloudflare DoH, bypassing system DNS."""
    for doh_url in [
        "https://cloudflare-dns.com/dns-query?name={}&type=A",
        "https://dns.google/resolve?name={}&type=A",
    ]:
        try:
            req = requests.get(
                doh_url.format(domain),
                headers={"Accept": "application/dns-json"},
                timeout=5,
                impersonate=False,
            )
            data = json.loads(req.text)
            for ans in data.get("Answer", []):
                if ans["type"] == 1:
                    ip = ans["data"]
                    if not ip.startswith(SUSPICIOUS_IPS):
                        return ip
        except Exception:
            continue
    return None


def _is_dns_hijacked():
    """Check if system DNS is returning hijacked IPs."""
    try:
        ips = socket.getaddrinfo("www.avito.ru", 443)
        ip = ips[0][4][0]
        return ip.startswith(SUSPICIOUS_IPS)
    except Exception:
        return False


def _resolve_avito():
    """Resolve Avito IP using DoH if system DNS is hijacked."""
    global AVITO_REAL_IPS
    if AVITO_REAL_IPS:
        return AVITO_REAL_IPS[0]

    if _is_dns_hijacked():
        logger.info("System DNS hijacked for Avito, using DoH resolution")
        ip = _resolve_via_doh("www.avito.ru")
        if ip:
            AVITO_REAL_IPS.append(ip)
            logger.info(f"Avito real IP: {ip}")
            return ip

    # System DNS might be OK, try normal resolution
    try:
        ips = socket.getaddrinfo("www.avito.ru", 443)
        ip = ips[0][4][0]
        AVITO_REAL_IPS.append(ip)
        return ip
    except Exception:
        return None


def _patch_socket(host: str, real_ip: str):
    """Monkey-patch socket to resolve host to real_ip."""
    original = socket.getaddrinfo

    def patched(h, port, family=0, type_=0, proto=0, flags=0):
        if h == host:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (real_ip, port))]
        return original(h, port, family, type_, proto, flags)

    socket.getaddrinfo = patched
    return original


async def parse_avito_prices_cffi(search_query: str, max_listings: int = MAX_LISTINGS, component_type: str = "") -> list[float] | object:
    """Parse Avito prices using curl_cffi with DNS bypass."""
    prices = []

    real_ip = _resolve_avito()
    if not real_ip:
        logger.error("Cannot resolve Avito IP")
        from backend.parser import BOOTCACHE_SENTINEL
        return BOOTCACHE_SENTINEL

    # Patch socket if DNS is hijacked
    restore = None
    if _is_dns_hijacked():
        restore = _patch_socket("www.avito.ru", real_ip)

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }

    kwargs = dict(
        headers=headers,
        impersonate="chrome120",
        timeout=20,
    )
    if PROXY_URL:
        kwargs["proxies"] = {"http": PROXY_URL, "https": PROXY_URL}

    encoded_query = quote(search_query)
    url = f"{AVITO_BASE_URL}/all?q={encoded_query}"

    try:
        r = requests.get(url, **kwargs)

        if r.status_code == 429 or r.status_code == 403:
            logger.warning(f"Avito blocked request (HTTP {r.status_code}) for: {search_query}")
            from backend.parser import BOOTCACHE_SENTINEL
            return BOOTCACHE_SENTINEL

        if r.status_code != 200:
            logger.warning(f"Avito returned {r.status_code} for: {search_query}")
            return prices

        text = r.text

        # Try multiple price extraction strategies — returns (title, price) tuples
        listings = _extract_listings_from_html(text, max_listings)
        prices = filter_listings(listings, component_type, search_query) if component_type else [p for _, p in listings]
        logger.info(f"Found {len(listings)} listings → {len(prices)} relevant")

    except Exception as e:
        logger.error(f"Parse error for '{search_query}': {e}")
    finally:
        if restore:
            socket.getaddrinfo = restore

    return prices


def _extract_listings_from_html(html: str, max_listings: int) -> list[tuple[str, float]]:
    """Extract listing titles and prices from Avito HTML page.
    Returns list of (title, price) tuples.
    """
    listings = []

    # Strategy 1: Find items with data-marker="item"
    # Match <div data-marker="item"> ... <h3 itemprop="name">TITLE</h3> ... <meta itemprop="price" content="PRICE">
    item_pattern = r'<div[^>]*data-marker="item"[^>]*>(.*?)</div>\s*(?=<div[^>]*data-marker="item"|$)'
    items = re.findall(r'<div[^>]*data-marker="item"[^>]*>.*?</div>\s*</div>', html, re.DOTALL)
    if not items:
        items = re.findall(r'<div[^>]*data-marker="item"[^>]*>.*?<meta\s+itemprop="price"[^>]*>', html, re.DOTALL)

    for item_html in items[:max_listings]:
        try:
            title = ""
            title_m = re.search(r'itemprop="name"[^>]*>(.*?)</', item_html)
            if title_m:
                title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip()

            price_m = re.search(r'<meta\s+itemprop="price"\s+content="(\d+)"', item_html)
            if price_m:
                p = float(price_m.group(1))
                if 100 <= p <= 10_000_000:
                    listings.append((title, p))
                    continue

            # Try alt price patterns
            price_m2 = re.search(r'<meta\s+content="(\d+)"\s+itemprop="price"', item_html)
            if price_m2:
                p = float(price_m2.group(1))
                if 100 <= p <= 10_000_000:
                    listings.append((title, p))
        except Exception:
            pass

    if listings:
        return listings

    # Strategy 2: Find all meta[itemprop="price"] directly
    for m in re.finditer(r'<meta\s+itemprop="price"\s+content="(\d+)"', html):
        try:
            p = float(m.group(1))
            if 100 <= p <= 10_000_000:
                listings.append(("", p))
        except ValueError:
            pass
        if len(listings) >= max_listings:
            break

    if listings:
        return listings

    # Strategy 3: Regex for price patterns in text
    seen_prices = set()
    for pat in [r'(\d[\d\s]{1,6}\d)\s*[₽pр]', r'(\d[\d\s]{1,6}\d)\s*руб']:
        for m in re.finditer(pat, html):
            try:
                clean = m.group(1).replace("\u00a0", "").replace(" ", "")
                p = float(clean)
                if 100 <= p <= 10_000_000 and p not in seen_prices:
                    seen_prices.add(p)
                    listings.append(("", p))
            except ValueError:
                pass
            if len(listings) >= max_listings:
                break
        if len(listings) >= 3:
            break

    return listings[:max_listings]
