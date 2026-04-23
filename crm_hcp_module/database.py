import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Database Connection (SQLite for local testing, can easily swap to PostgreSQL connection string)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./crm.db")

# Engine setup with SQLite specific threading constraint disabled
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

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
    hcp_name = Column(String, index=True, nullable=False)
    interaction_type = Column(String, nullable=True)
    date = Column(String, nullable=True)
    time = Column(String, nullable=True)
    attendees = Column(String, nullable=True)
    topics_discussed = Column(String, nullable=True)
    materials_shared = Column(String, nullable=True)
    samples_distributed = Column(String, nullable=True)
    sentiment = Column(String, nullable=True)
    outcomes = Column(String, nullable=True)
    follow_up_actions = Column(String, nullable=True)
    
    # Store AI suggested follow ups as a comma separated string to keep the SQLite schema simple
    ai_suggested_follow_ups = Column(String, nullable=True)

# Dependency injection for FastAPI
def get_db():
    """Yields a database session and safely closes it when done"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
