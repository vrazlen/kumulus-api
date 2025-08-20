# src/kumulus_consultant/db.py
from sqlalchemy import create_engine
from .config.settings import settings

# Create a single, application-wide engine with a connection pool.
# This engine will be imported by any part of the app that needs DB access.
# The configuration values are drawn from our Pydantic settings object.
# [cite: 86, 87, 88]
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,           # Number of connections to keep open in the pool [cite: 91]
    max_overflow=10,       # Number of additional connections allowed beyond pool_size [cite: 91]
    pool_recycle=3600,     # Recycle connections after 1 hour to prevent stale connections [cite: 91]
    pool_timeout=30,       # Time to wait for a connection before timing out [cite: 91]
)