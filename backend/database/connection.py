"""
Database connection module for SQLite and ChromaDB.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Union

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, OllamaEmbeddingFunction
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Collection

# Database paths
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data"
SQLITE_DB_PATH = DATA_DIR / "lamb-kb-server.db"
CHROMA_DB_PATH = "/app/data"

# Ensure the directories exist
DATA_DIR.mkdir(exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(exist_ok=True)

# Create SQLite engine
SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)


def get_embedding_function_by_params(vendor: str, model_name: str, api_key: str = "", api_endpoint: str = ""):
    """Get an embedding function based on vendor and model parameters."""
    vendor = vendor.lower()
    
    if vendor in ("ollama", "local"):
        return OllamaEmbeddingFunction(
            url=api_endpoint or "http://localhost:11434",
            model_name=model_name
        )
    
    elif vendor == "openai":
        kwargs = {"api_key": api_key, "model_name": model_name}
        if api_endpoint:
            # If api_endpoint ends with '/embeddings', strip it for OpenAIEmbeddingFunction
            if api_endpoint.endswith("/embeddings"):
                api_endpoint = api_endpoint[:-len("/embeddings")]
            kwargs["api_base"] = api_endpoint
        return OpenAIEmbeddingFunction(**kwargs)
    
    else:
        raise ValueError(f"Unsupported embedding vendor: {vendor}")


def get_db() -> Session:
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_chroma_client() -> chromadb.PersistentClient:
    """Get the ChromaDB client."""
    return chroma_client


def init_sqlite_db() -> None:
    """Initialize the SQLite database."""
    Base.metadata.create_all(bind=engine)


def get_embedding_function(collection_id_or_obj: Union[int, Collection, Dict[str, Any]]) -> Callable:
    """Get the embedding function for a collection by its ID or Collection object."""
    db = next(get_db())
    
    try:
        # Handle dict case first
        if isinstance(collection_id_or_obj, dict):
            if 'embeddings_model' in collection_id_or_obj:
                embedding_config = collection_id_or_obj['embeddings_model']
                return get_embedding_function_by_params(
                    embedding_config.get("vendor"),
                    embedding_config.get("model"),
                    embedding_config.get("apikey"),
                    embedding_config.get("api_endpoint")
                )
            collection_id = collection_id_or_obj.get('id')
            if not collection_id:
                raise ValueError("Collection dictionary must contain an 'id' field")
            collection = db.query(Collection).filter(Collection.id == collection_id).first()
        # Handle Collection object case
        elif isinstance(collection_id_or_obj, Collection):
            collection = collection_id_or_obj
        # Handle integer ID case
        elif isinstance(collection_id_or_obj, int):
            collection = db.query(Collection).filter(Collection.id == collection_id_or_obj).first()
        else:
            raise ValueError(f"Expected Collection object, dictionary or ID")
            
        if not collection:
            raise ValueError(f"Collection not found")
            
        # Extract embedding configuration
        embedding_config = json.loads(collection.embeddings_model) if isinstance(collection.embeddings_model, str) else collection.embeddings_model
        
        # Use the helper function to get the actual embedding function
        return get_embedding_function_by_params(
            embedding_config.get("vendor"),
            embedding_config.get("model"),
            embedding_config.get("apikey"),
            embedding_config.get("api_endpoint")
        )
        
    finally:
        db.close()


def check_sqlite_schema() -> bool:
    """Check if the SQLite database schema is compatible."""
    inspector = inspect(engine)
    
    if "collections" not in inspector.get_table_names():
        return True
    
    collection_columns = {col["name"] for col in inspector.get_columns("collections")}
    required_columns = {"id", "name", "description", "creation_date", "owner", "visibility", "embeddings_model"}
    
    return required_columns.issubset(collection_columns)


def init_databases() -> Dict[str, Any]:
    """Initialize all databases and perform sanity checks."""
    status = {
        "sqlite_initialized": False,
        "sqlite_schema_valid": False,
        "chromadb_initialized": False,
        "errors": []
    }
    
    try:
        status["sqlite_schema_valid"] = check_sqlite_schema()
        
        if not status["sqlite_schema_valid"]:
            status["errors"].append("SQLite schema is not compatible")
        
        init_sqlite_db()
        status["sqlite_initialized"] = True
        
        status["chromadb_collections"] = len(chroma_client.list_collections())
        status["chromadb_initialized"] = True
        
    except Exception as e:
        status["errors"].append(f"Error initializing databases: {str(e)}")
    
    return status