import logging
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.orm import Session as DBSession

from backend.config import DATABASE_URL, COMPONENT_TYPES

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ComponentType(Base):
    __tablename__ = "component_types"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)


class Build(Base):
    __tablename__ = "builds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    status = Column(String, default="pending")
    progress = Column(Integer, default=0)
    total_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    items = relationship("BuildItem", back_populates="build", cascade="all, delete-orphan")


class BuildItem(Base):
    __tablename__ = "build_items"

    id = Column(Integer, primary_key=True, index=True)
    build_id = Column(Integer, ForeignKey("builds.id"), nullable=False)
    component_type_id = Column(Integer, ForeignKey("component_types.id"), nullable=False)
    search_query = Column(String, nullable=False)
    status = Column(String, default="pending")
    price_cache_id = Column(Integer, ForeignKey("price_cache.id"), nullable=True)
    is_hidden = Column(Integer, default=0)
    sort_order = Column(Integer, default=0)

    build = relationship("Build", back_populates="items")
    component_type = relationship("ComponentType")
    price_cache = relationship("PriceCache")


class BuildPhoto(Base):
    __tablename__ = "build_photos"

    id = Column(Integer, primary_key=True, index=True)
    build_id = Column(Integer, ForeignKey("builds.id"), nullable=False)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    build = relationship("Build", backref="photos")


class PriceCache(Base):
    __tablename__ = "price_cache"

    id = Column(Integer, primary_key=True, index=True)
    component_type_id = Column(Integer, ForeignKey("component_types.id"), nullable=False)
    search_query = Column(String, nullable=False)
    avg_price = Column(Float, nullable=True)
    median_price = Column(Float, nullable=True)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    listings_count = Column(Integer, default=0)
    listings_raw = Column(Integer, default=0)
    parsed_at = Column(DateTime, default=datetime.utcnow)


class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    component_type_id = Column(Integer, ForeignKey("component_types.id"), nullable=False)
    search_query = Column(String, nullable=False)
    avg_price = Column(Float, nullable=True)
    median_price = Column(Float, nullable=True)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    parsed_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for slug, name in COMPONENT_TYPES.items():
            existing = db.query(ComponentType).filter(ComponentType.slug == slug).first()
            if not existing:
                db.add(ComponentType(slug=slug, name=name))
        db.commit()
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
