"""
Database models for the Lamb Knowledge Base Server.

This module defines SQLAlchemy models for the application's database schema.
"""

# Python Libraries
import datetime
import json
from enum import Enum
from typing import Any, Dict

# Third-Party Libraries
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Enum as SQLAlchemyEnum

Base = declarative_base()


class Visibility(str, Enum):
    """Enum for collection visibility states."""
    PRIVATE = "private"
    PUBLIC = "public"


class FileStatus(str, Enum):
    """Enum for file status states."""
    COMPLETED = "completed"
    PROCESSING = "processing"
    FAILED = "failed"
    DELETED = "deleted"


class Collection(Base):
    """Model representing a knowledge base collection.
    
    Each collection has an associated ChromaDB collection.
    """
    __tablename__ = "collections"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    creation_date = Column(DateTime, default=datetime.datetime.utcnow)
    owner = Column(String(255), nullable=False, index=True)
    visibility = Column(SQLAlchemyEnum(Visibility), default=Visibility.PRIVATE, nullable=False)
    embeddings_model = Column(JSON, nullable=False, 
                              default=lambda: json.dumps({
                                  "model": "sentence-transformers/all-MiniLM-L6-v2",
                                  "endpoint": None,
                                  "apikey": None
                              }))
    chromadb_uuid = Column(String(36), nullable=True, unique=True, index=True)
    
    __table_args__ = (
        UniqueConstraint('name', 'owner', name='uix_collection_name_owner'),
    )
    
    def __repr__(self):
        return f"<Collection id={self.id}, name={self.name}, owner={self.owner}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "owner": self.owner,
            "visibility": self.visibility.value,
            "embeddings_model": json.loads(self.embeddings_model) if isinstance(self.embeddings_model, str) else self.embeddings_model,
            "chromadb_uuid": self.chromadb_uuid
        }


class FileRegistry(Base):
    """Model representing files ingested into collections.
    
    This tracks each file added to a collection, along with the ingestion parameters
    and status.
    """
    __tablename__ = "file_registry"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_url = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=True)
    plugin_name = Column(String(100), nullable=False)
    plugin_params = Column(JSON, nullable=False)
    status = Column(SQLAlchemyEnum(FileStatus), default=FileStatus.COMPLETED, nullable=False)
    document_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)
    owner = Column(String(255), nullable=False, index=True)
    
    def __repr__(self):
        return f"<FileRegistry id={self.id}, collection_id={self.collection_id}, filename={self.original_filename}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "file_url": self.file_url,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "plugin_name": self.plugin_name,
            "plugin_params": json.loads(self.plugin_params) if isinstance(self.plugin_params, str) else self.plugin_params,
            "status": self.status.value,
            "document_count": self.document_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "owner": self.owner
        }
