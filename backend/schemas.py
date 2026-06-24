from pydantic import BaseModel
from typing import Optional


class ItemInput(BaseModel):
    component_type: str
    search_query: str


class BuildCreate(BaseModel):
    name: str = "Игровая сборка"
    items: list[ItemInput]


class PriceStatsResponse(BaseModel):
    avg: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    listings: int = 0
    listings_raw: int = 0
    source: str = ""


class ItemResponse(BaseModel):
    id: int
    component_type: str
    component_name: str
    search_query: str
    status: str
    price: Optional[PriceStatsResponse] = None


class BuildResponse(BaseModel):
    id: int
    name: str
    status: str
    progress: int
    total_price: Optional[float] = None
    items: list[ItemResponse] = []
    created_at: str


class JobStatusResponse(BaseModel):
    status: str
    progress: int
    items_total: int
    items_done: int
    items: list[ItemResponse] = []
