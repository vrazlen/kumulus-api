from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Numeric, DateTime, text
from typing import List
from .models import Settlement as SettlementModel

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)
AsyncDBSession = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Define the SQLAlchemy model to match our database table
class SettlementDB(Base):
    __tablename__ = 'settlements'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    kumuh_score = Column(Numeric)
    # We can ignore the geometry and other fields for now
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


app = FastAPI()

# Dependency to get a DB session
async def get_db() -> AsyncSession:
    async with AsyncDBSession() as session:
        yield session

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}

@app.get("/settlements/", response_model=List[SettlementModel])
async def get_settlements(db: AsyncSession = Depends(get_db)):
    """
    Retrieves a list of all settlements from the database.
    """
    result = await db.execute(text("SELECT id, name, kumuh_score, created_at FROM settlements;"))
    settlements = result.fetchall()
    return settlements