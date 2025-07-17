# Python Libraries
from typing import Any, Dict

# Third-Party Libraries
from pydantic import BaseModel, Field

# Response models
class HealthResponse(BaseModel):
    """Model for health check responses"""
    status: str = Field(..., description="Status of the server", example="ok")
    version: str = Field(..., description="Server version", example="0.1.0")

class MessageResponse(BaseModel):
    """Model for basic message responses"""
    message: str = Field(..., description="Response message", example="Hello World from the Lamb Knowledge Base Server!")

class DatabaseStatusResponse(BaseModel):
    """Model for database status response"""
    sqlite_status: Dict[str, Any] = Field(..., description="Status of SQLite database")
    chromadb_status: Dict[str, Any] = Field(..., description="Status of ChromaDB database")
    collections_count: int = Field(..., description="Number of collections") 