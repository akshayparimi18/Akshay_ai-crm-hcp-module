import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Load environment variables
load_dotenv()

# Database Connection — pulled securely from .env
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the .env file. Please add your PostgreSQL connection string.")

# Engine setup for PostgreSQL (pool_pre_ping keeps connections alive)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# ---------------------------------------------------------
# ORM Models for the CRM
# ---------------------------------------------------------
class Interaction(Base):
    """Stores the main interaction log with the HCP using the expanded schema."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    hcp_name = Column(String(255), index=True, nullable=False)
    interaction_type = Column(String(100), nullable=True)
    date = Column(String(50), nullable=True)
    time = Column(String(50), nullable=True)
    attendees = Column(Text, nullable=True)
    topics_discussed = Column(Text, nullable=True)
    materials_shared = Column(Text, nullable=True)
    samples_distributed = Column(String(100), nullable=True)
    sentiment = Column(String(50), nullable=True)
    outcomes = Column(Text, nullable=True)
    follow_up_actions = Column(Text, nullable=True)
    
    # Store AI suggested follow ups as JSON string
    ai_suggested_follow_ups = Column(Text, nullable=True)

# Dependency injection for FastAPI
def get_db():
    """Yields a database session and safely closes it when done"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
