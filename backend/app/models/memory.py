from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import relationship
from ..core.database import Base

class Memory(Base):
    __tablename__ = "memories"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    text = Column(String(1000), nullable=False)
    embedding_ref = Column(String(100), index=True)  # key in vector store
    meta = Column("metadata", JSON, nullable=True)  # DB column 'metadata' but attr 'meta'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="memories")
