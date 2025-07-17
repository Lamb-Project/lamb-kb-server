# Python Libraries
from typing import Any, Dict, Optional

# Third-Party Libraries
from pydantic import BaseModel, Field

# Import file registry schemas
class FileRegistryResponse(BaseModel):
    """Model for file registry entry response"""
    id: int = Field(..., description="ID of the file registry entry")
    collection_id: int = Field(..., description="ID of the collection")
    original_filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path to the file on the server")
    file_url: str = Field(..., description="URL to access the file")
    file_size: int = Field(..., description="Size of the file in bytes")
    content_type: Optional[str] = Field(None, description="MIME type of the file")
    plugin_name: str = Field(..., description="Name of the ingestion plugin used")
    plugin_params: Dict[str, Any] = Field(..., description="Parameters used for ingestion")
    status: str = Field(..., description="Status of the file")
    document_count: int = Field(..., description="Number of documents created from this file")
    created_at: str = Field(..., description="Timestamp when the file was added")
    updated_at: str = Field(..., description="Timestamp when the file record was last updated")
    owner: str = Field(..., description="Owner of the file") 