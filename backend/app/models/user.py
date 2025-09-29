from sqlalchemy import Column, Integer, String, DateTime, func
from ..core.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    display_name = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
