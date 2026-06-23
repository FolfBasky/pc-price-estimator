import os

CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "7"))
AVITO_BASE_URL = "https://www.avito.ru"
PARSER_DELAY_MIN = int(os.getenv("PARSER_DELAY_MIN", "2"))
PARSER_DELAY_MAX = int(os.getenv("PARSER_DELAY_MAX", "5"))
MAX_LISTINGS = int(os.getenv("MAX_LISTINGS", "100"))
HEADLESS = os.getenv("PARSER_HEADLESS", "true").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pc_prices.db")
PROXY_URL = os.getenv("PROXY_URL", "")

COMPONENT_TYPES = {
    "cpu": "Процессор",
    "gpu": "Видеокарта",
    "ram": "Оперативная память",
    "motherboard": "Материнская плата",
    "storage": "Накопитель (SSD/HDD)",
    "psu": "Блок питания",
    "case": "Корпус",
    "cooler": "Охлаждение",
    "other": "Другое",
}
