# Start PC Price Estimator
Write-Host "Starting PC Price Estimator..." -ForegroundColor Green
Write-Host "Open http://localhost:8000 in your browser" -ForegroundColor Cyan
Write-Host ""
Write-Host "Optional env vars:" -ForegroundColor Yellow
Write-Host "  PROXY_URL        - http://user:pass@host:port  (bypass Avito block)" -ForegroundColor Gray
Write-Host "  CACHE_TTL_DAYS   - default: 7" -ForegroundColor Gray
Write-Host "  PARSER_HEADLESS  - false to see browser window" -ForegroundColor Gray
Write-Host ""

$env:PYTHONPATH = "D:\pc-price-estimator"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
