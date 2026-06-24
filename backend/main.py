import logging
import os
import random
import re
import string
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

from backend.database import init_db, SessionLocal, Build, BuildItem, ComponentType, PriceCache, PriceHistory, BuildPhoto
from backend.schemas import BuildCreate, BuildResponse, ItemResponse, PriceStatsResponse, JobStatusResponse
from backend.price_engine import process_build, calc_stats
from backend.config import COMPONENT_TYPES
from backend.query_cleaner import simplify_query
from backend.ai_parser import load_model

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Start AI model loading in background (does not block startup)
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield

app = FastAPI(title="PC Price Estimator", lifespan=lifespan)

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Serve static files (favicon, etc.)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---- Page routes ----

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"component_types": COMPONENT_TYPES}
    )


@app.get("/builds", response_class=HTMLResponse)
async def builds_page(request: Request):
    db = SessionLocal()
    try:
        builds = db.query(Build).order_by(Build.created_at.desc()).all()
        return templates.TemplateResponse(
            request, "builds.html", {"builds": builds}
        )
    finally:
        db.close()


@app.get("/builds/{build_id}", response_class=HTMLResponse)
async def build_page(request: Request, build_id: int):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            return HTMLResponse("Сборка не найдена", status_code=404)

        items = db.query(BuildItem).filter(BuildItem.build_id == build_id).order_by(BuildItem.sort_order, BuildItem.id).all()
        items_data = []
        items_hidden = []
        items_done = 0
        for item in items:
            ct = db.query(ComponentType).filter(ComponentType.id == item.component_type_id).first()
            price_data = None
            if item.price_cache_id:
                pc = db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).first()
                if pc:
                    price_data = {
                        "avg": float(pc.avg_price) if pc.avg_price else None,
                        "median": float(pc.median_price) if pc.median_price else None,
                        "min": float(pc.min_price) if pc.min_price else None,
                        "max": float(pc.max_price) if pc.max_price else None,
                        "listings": pc.listings_count,
                        "listings_raw": pc.listings_raw or pc.listings_count,
                        "source": "cache" if pc.parsed_at else "live",
                    }
            if item.status == "done":
                items_done += 1
            entry = {
                "id": item.id,
                "component_name": ct.name if ct else "?",
                "search_query": item.search_query,
                "status": item.status,
                "price": price_data,
                "is_hidden": item.is_hidden,
            }
            if item.is_hidden:
                items_hidden.append(entry)
            else:
                items_data.append(entry)

        photos = db.query(BuildPhoto).filter(BuildPhoto.build_id == build_id).order_by(BuildPhoto.uploaded_at.desc()).all()
        photos_data = [{"id": p.id, "filename": p.filename, "original_name": p.original_name} for p in photos]

        return templates.TemplateResponse(
            request, "build.html",
            {
                "request": request,
                "build": build,
                "items_data": items_data,
                "items_hidden": items_hidden,
                "items_total": len(items),
                "items_done": items_done,
                "photos": photos_data,
                "component_types": COMPONENT_TYPES,
            },
        )
    finally:
        db.close()


@app.get("/builds/{build_id}/print", response_class=HTMLResponse)
async def build_print_page(request: Request, build_id: int):
    """Print-friendly page with table, photos and total."""
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            return HTMLResponse("Сборка не найдена", status_code=404)

        items = db.query(BuildItem).filter(
            BuildItem.build_id == build_id,
            BuildItem.is_hidden == 0,
        ).order_by(BuildItem.sort_order, BuildItem.id).all()

        items_data = []
        total = 0.0
        for item in items:
            ct = db.query(ComponentType).filter(ComponentType.id == item.component_type_id).first()
            price_data = None
            if item.price_cache_id:
                pc = db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).first()
                if pc and pc.median_price:
                    price_data = {
                        "median": float(pc.median_price),
                        "avg": float(pc.avg_price) if pc.avg_price else None,
                    }
                    total += float(pc.median_price)
            items_data.append({
                "component_name": ct.name if ct else "?",
                "search_query": item.search_query,
                "price": price_data,
            })

        photos = db.query(BuildPhoto).filter(BuildPhoto.build_id == build_id).order_by(BuildPhoto.uploaded_at).all()

        return templates.TemplateResponse(
            request, "print.html",
            {
                "request": request,
                "build": build,
                "items_data": items_data,
                "total": round(total, 2),
                "photos": photos,
            },
        )
    finally:
        db.close()


# ---- API routes ----

@app.post("/api/builds")
def create_build(data: BuildCreate):
    db = SessionLocal()
    try:
        build = Build(name=data.name, status="pending")
        db.add(build)
        db.commit()
        db.refresh(build)

        for i, item_in in enumerate(data.items):
            ct = db.query(ComponentType).filter(ComponentType.slug == item_in.component_type).first()
            if not ct:
                continue
            simplified = simplify_query(item_in.search_query)
            item = BuildItem(
                build_id=build.id,
                component_type_id=ct.id,
                search_query=simplified,
                status="pending",
                sort_order=i,
            )
            db.add(item)
        db.commit()

        import threading
        t = threading.Thread(target=lambda: __import__('asyncio').run(process_build(build.id)), daemon=True)
        t.start()

        return {"id": build.id, "status": "pending"}
    finally:
        db.close()


@app.get("/api/builds")
def list_builds():
    db = SessionLocal()
    try:
        builds = db.query(Build).order_by(Build.created_at.desc()).all()
        result = []
        for b in builds:
            result.append({
                "id": b.id,
                "name": b.name,
                "status": b.status,
                "progress": b.progress,
                "total_price": float(b.total_price) if b.total_price else None,
                "created_at": b.created_at.isoformat() if b.created_at else None,
            })
        return result
    finally:
        db.close()


@app.get("/api/builds/{build_id}")
def get_build(build_id: int):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            raise HTTPException(404, "Build not found")

        items = db.query(BuildItem).filter(BuildItem.build_id == build_id).all()
        items_resp = []
        for item in items:
            ct = db.query(ComponentType).filter(ComponentType.id == item.component_type_id).first()
            price = None
            if item.price_cache_id:
                pc = db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).first()
                if pc:
                    price = PriceStatsResponse(
                        avg=float(pc.avg_price) if pc.avg_price else None,
                        median=float(pc.median_price) if pc.median_price else None,
                        min=float(pc.min_price) if pc.min_price else None,
                        max=float(pc.max_price) if pc.max_price else None,
                        listings=pc.listings_count,
                        listings_raw=pc.listings_raw or pc.listings_count,
                        source="cache",
                    )
            items_resp.append(ItemResponse(
                id=item.id,
                component_type=ct.slug if ct else "",
                component_name=ct.name if ct else "",
                search_query=item.search_query,
                status=item.status,
                price=price,
            ))
        return BuildResponse(
            id=build.id,
            name=build.name,
            status=build.status,
            progress=build.progress,
            total_price=float(build.total_price) if build.total_price else None,
            items=items_resp,
            created_at=build.created_at.isoformat() if build.created_at else "",
        )
    finally:
        db.close()


@app.get("/api/builds/{build_id}/status")
def get_build_status(build_id: int):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            raise HTTPException(404)
        items = db.query(BuildItem).filter(BuildItem.build_id == build_id).order_by(BuildItem.sort_order, BuildItem.id).all()
        items_done = sum(1 for i in items if i.status in ("done", "error"))
        item_responses = []
        for item in items:
            ct = db.query(ComponentType).filter(ComponentType.id == item.component_type_id).first()
            price = None
            if item.price_cache_id:
                pc = db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).first()
                if pc:
                    price = PriceStatsResponse(
                        avg=float(pc.avg_price) if pc.avg_price else None,
                        median=float(pc.median_price) if pc.median_price else None,
                        min=float(pc.min_price) if pc.min_price else None,
                        max=float(pc.max_price) if pc.max_price else None,
                        listings=pc.listings_count,
                        listings_raw=pc.listings_raw or pc.listings_count,
                        source="cache",
                    )
            item_responses.append(ItemResponse(
                id=item.id,
                component_type=ct.slug if ct else "",
                component_name=ct.name if ct else "",
                search_query=item.search_query,
                status=item.status,
                price=price,
            ))
        return JobStatusResponse(
            status=build.status,
            progress=build.progress,
            items_total=len(items),
            items_done=items_done,
            items=item_responses,
        )
    finally:
        db.close()


@app.post("/api/builds/{build_id}/refresh")
def refresh_build(build_id: int):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            raise HTTPException(404)

        items = db.query(BuildItem).filter(
            BuildItem.build_id == build_id,
            BuildItem.is_hidden == 0,
        ).all()
        for item in items:
            if item.price_cache_id:
                # Delete the cached entry so fresh parse is forced
                db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).delete()
            item.status = "pending"
            item.price_cache_id = None

        build.status = "pending"
        build.progress = 0
        build.total_price = None
        db.commit()

        import threading
        t = threading.Thread(target=lambda: __import__('asyncio').run(process_build(build_id)), daemon=True)
        t.start()

        return RedirectResponse(url=f"/builds/{build_id}", status_code=303)
    finally:
        db.close()


class ManualPriceInput(BaseModel):
    median_price: float
    avg_price: float | None = None
    min_price: float | None = None
    max_price: float | None = None


@app.post("/api/builds/{build_id}/items/{item_id}/price")
def set_manual_price(build_id: int, item_id: int, data: ManualPriceInput):
    db = SessionLocal()
    try:
        item = db.query(BuildItem).filter(
            BuildItem.id == item_id,
            BuildItem.build_id == build_id,
        ).first()
        if not item:
            raise HTTPException(404)

        cache = PriceCache(
            component_type_id=item.component_type_id,
            search_query=item.search_query,
            avg_price=data.avg_price or data.median_price,
            median_price=data.median_price,
            min_price=data.min_price or data.median_price,
            max_price=data.max_price or data.median_price,
            listings_count=1,
            parsed_at=datetime.utcnow(),
        )
        db.add(cache)
        db.commit()
        db.refresh(cache)

        item.price_cache_id = cache.id
        item.status = "done"
        db.commit()

        _recalc_total(db, build_id)

        return {"status": "ok", "total_price": _get_total(db, build_id)}
    finally:
        db.close()


@app.post("/api/builds/{build_id}/items/{item_id}/toggle")
def toggle_item_hidden(build_id: int, item_id: int):
    db = SessionLocal()
    try:
        item = db.query(BuildItem).filter(
            BuildItem.id == item_id,
            BuildItem.build_id == build_id,
        ).first()
        if not item:
            raise HTTPException(404)
        item.is_hidden = 0 if item.is_hidden else 1
        db.commit()
        _recalc_total(db, build_id)
        return {"status": "ok", "is_hidden": bool(item.is_hidden)}
    finally:
        db.close()


@app.delete("/api/builds/{build_id}/items/{item_id}")
def delete_item(build_id: int, item_id: int):
    db = SessionLocal()
    try:
        item = db.query(BuildItem).filter(
            BuildItem.id == item_id,
            BuildItem.build_id == build_id,
        ).first()
        if not item:
            raise HTTPException(404)
        db.delete(item)
        db.commit()
        _recalc_total(db, build_id)
        return {"status": "ok"}
    finally:
        db.close()


class AddItemInput(BaseModel):
    component_type: str
    search_query: str


@app.post("/api/builds/{build_id}/items")
def add_item(build_id: int, data: AddItemInput):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            raise HTTPException(404)
        ct = db.query(ComponentType).filter(ComponentType.slug == data.component_type).first()
        if not ct:
            raise HTTPException(400, "Invalid component type")

        max_order = db.query(BuildItem.sort_order).filter(
            BuildItem.build_id == build_id
        ).order_by(BuildItem.sort_order.desc()).first()
        next_order = (max_order[0] or 0) + 1 if max_order else 0

        simplified = simplify_query(data.search_query)
        item = BuildItem(
            build_id=build_id,
            component_type_id=ct.id,
            search_query=simplified,
            status="pending",
            sort_order=next_order,
        )
        db.add(item)
        db.commit()
        db.refresh(item)

        threading.Thread(target=lambda: __import__('asyncio').run(
            _parse_single_item(build_id, item.id)
        ), daemon=True).start()

        return {"id": item.id, "status": "pending"}
    finally:
        db.close()


class EditItemInput(BaseModel):
    search_query: str


@app.put("/api/builds/{build_id}/items/{item_id}")
def edit_item(build_id: int, item_id: int, data: EditItemInput):
    db = SessionLocal()
    try:
        item = db.query(BuildItem).filter(
            BuildItem.id == item_id,
            BuildItem.build_id == build_id,
        ).first()
        if not item:
            raise HTTPException(404)

        simplified = simplify_query(data.search_query)
        item.search_query = simplified
        item.status = "pending"

        # Clear cached price for this item
        if item.price_cache_id:
            pc = db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).first()
            if pc:
                db.delete(pc)
            item.price_cache_id = None

        build = db.query(Build).filter(Build.id == build_id).first()
        if build:
            build.status = "pending"
            build.progress = 0

        db.commit()

        threading.Thread(target=lambda: __import__('asyncio').run(
            _parse_single_item(build_id, item.id)
        ), daemon=True).start()

        return {"status": "ok"}
    finally:
        db.close()


# ---- Photo endpoints ----

@app.post("/api/builds/{build_id}/photos")
async def upload_photo(build_id: int, file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            raise HTTPException(404)

        ext = os.path.splitext(file.filename or "photo.jpg")[1] or ".jpg"
        name = f"{build_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}{ext}"
        path = os.path.join(UPLOAD_DIR, name)
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)

        photo = BuildPhoto(
            build_id=build_id,
            filename=name,
            original_name=file.filename or "photo",
        )
        db.add(photo)
        db.commit()
        db.refresh(photo)
        return {"id": photo.id, "filename": name, "original_name": photo.original_name}
    finally:
        db.close()


@app.get("/api/builds/{build_id}/photos")
def list_photos(build_id: int):
    db = SessionLocal()
    try:
        photos = db.query(BuildPhoto).filter(BuildPhoto.build_id == build_id).order_by(BuildPhoto.uploaded_at.desc()).all()
        return [{"id": p.id, "filename": p.filename, "original_name": p.original_name} for p in photos]
    finally:
        db.close()


@app.delete("/api/builds/{build_id}/photos/{photo_id}")
def delete_photo(build_id: int, photo_id: int):
    db = SessionLocal()
    try:
        photo = db.query(BuildPhoto).filter(
            BuildPhoto.id == photo_id,
            BuildPhoto.build_id == build_id,
        ).first()
        if not photo:
            raise HTTPException(404)
        path = os.path.join(UPLOAD_DIR, photo.filename)
        if os.path.exists(path):
            os.remove(path)
        db.delete(photo)
        db.commit()
        return {"status": "ok"}
    finally:
        db.close()


# ---- AI status & parse ----

@app.get("/api/ai-status")
def ai_status():
    """Return AI model loading status."""
    from backend.ai_parser import get_status
    return get_status()


class ParseDescriptionInput(BaseModel):
    text: str


@app.post("/api/parse-description")
def parse_description_endpoint(data: ParseDescriptionInput):
    """Parse a build description text using AI."""
    from backend.ai_parser import parse_description, is_ready
    text = data.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")
    if len(text) > 5000:
        raise HTTPException(400, "Text too long (max 5000 characters)")
    if re.search(r'[\x00-\x08\x0E-\x1F]', text):
        raise HTTPException(400, "Binary content not supported — text only")
    if re.match(r'^[\{A-F0-9\}-]+\.(png|jpg|jpeg|gif|webp|bmp)$', text, re.IGNORECASE):
        raise HTTPException(400, "Image files are not supported — paste a text description instead")
    items = parse_description(text)
    return {"items": items, "ai_ready": is_ready()}


# ---- Export ----

@app.get("/api/builds/{build_id}/export/txt")
def export_text(build_id: int):
    db = SessionLocal()
    try:
        build = db.query(Build).filter(Build.id == build_id).first()
        if not build:
            raise HTTPException(404)

        items = db.query(BuildItem).filter(
            BuildItem.build_id == build_id,
            BuildItem.is_hidden == 0,
        ).order_by(BuildItem.sort_order, BuildItem.id).all()

        lines = [
            "=" * 50,
            f"  СБОРКА: {build.name}",
            f"  Дата: {build.created_at.strftime('%d.%m.%Y %H:%M') if build.created_at else '-'}",
            "=" * 50,
            "",
        ]
        total = 0.0
        for item in items:
            ct = db.query(ComponentType).filter(ComponentType.id == item.component_type_id).first()
            name = ct.name if ct else "?"
            if item.price_cache_id:
                pc = db.query(PriceCache).filter(PriceCache.id == item.price_cache_id).first()
                price_str = f"{float(pc.median_price):,.0f} ₽".replace(",", " ") if pc and pc.median_price else "—"
                if pc and pc.median_price:
                    total += float(pc.median_price)
            else:
                price_str = "—"
            lines.append(f"  {name:25s} | {item.search_query:40s} | {price_str:>12s}")

        lines += [
            "",
            "-" * 50,
            f"  ИТОГО: {total:,.0f} ₽".replace(",", " "),
            "-" * 50,
        ]
        return PlainTextResponse("\n".join(lines), media_type="text/plain; charset=utf-8",
                                 headers={"Content-Disposition": f"attachment; filename=build_{build_id}.txt"})
    finally:
        db.close()


# ---- Helpers ----

def _recalc_total(db, build_id: int):
    build = db.query(Build).filter(Build.id == build_id).first()
    if not build:
        return
    items = db.query(BuildItem).filter(
        BuildItem.build_id == build_id,
        BuildItem.is_hidden == 0,
    ).all()
    total = 0.0
    for it in items:
        if it.price_cache_id:
            pc = db.query(PriceCache).filter(PriceCache.id == it.price_cache_id).first()
            if pc and pc.median_price:
                total += float(pc.median_price)
    build.total_price = round(total, 2) if total > 0 else None
    db.commit()


def _get_total(db, build_id: int):
    build = db.query(Build).filter(Build.id == build_id).first()
    return float(build.total_price) if build and build.total_price else 0


async def _parse_single_item(build_id: int, item_id: int):
    """Parse a single item that was added to an existing build."""
    from backend.price_engine import parse_single_item as _psi
    await _psi(build_id, item_id)
