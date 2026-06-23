# PC Price Estimator

Estimates PC component prices by parsing Avito listings via Selenium.

## Quick Start

```powershell
pip install -r requirements.txt
.\start.ps1
```

Open http://localhost:8000

## AI Model

This project uses Qwen2.5-3B-Instruct for listing verification. Download it manually:

```powershell
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/models--Qwen--Qwen2.5-3B-Instruct
```

Or place the model folder in `models/models--Qwen--Qwen2.5-3B-Instruct/` (snapshot format from HuggingFace cache).

The parser works without the AI model — listing filtering falls back to rule-based checks.

## Environment Variables

| Variable       | Description                           |
|---------------|---------------------------------------|
| PROXY_URL     | Proxy for Avito (http://user:pass@host:port) |
| CACHE_TTL_DAYS| Price cache expiry (default: 7)       |

## Project Structure

```
backend/
  main.py             FastAPI app & routes
  parser_selenium.py  Selenium Avito parser
  price_engine.py     Price calculation & IQR filtering
  listing_filter.py   Rule-based listing relevance filter
  ai_parser.py        AI model wrapper (Qwen2.5-3B)
  config.py           Configuration
  database.py         SQLite models
  schemas.py          Pydantic schemas
  templates/          Jinja2 HTML templates
models/               AI model files (download separately)
```

## Notes

- Requires Chrome for Selenium parsing
- Avito may block requests — use PROXY_URL or solve captcha in the opened browser window
