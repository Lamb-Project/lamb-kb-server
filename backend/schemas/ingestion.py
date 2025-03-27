"""
Schemas for document ingestion.

This module defines Pydantic schemas for document ingestion requests and responses.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union

from pydantic import BaseModel, Field, validator

from plugins.base import ChunkUnit


class IngestionPluginInfo(BaseModel):
    """Information about an ingestion plugin."""
    
    name: str = Field(..., description="Name of the plugin")
    description: str = Field(..., description="Description of the plugin")
    supported_file_types: List[str] = Field(..., description="File types supported by the plugin")
    parameters: Dict[str, Dict[str, Any]] = Field(..., description="Parameters accepted by the plugin")


class SimpleIngestParams(BaseModel):
    """Parameters for the simple_ingest plugin."""
    
    chunk_size: int = Field(1000, description="Size of each chunk")
    chunk_unit: ChunkUnit = Field(ChunkUnit.CHAR, description="Unit for chunking (char, word, line)")
    chunk_overlap: int = Field(200, description="Number of units to overlap between chunks")
    
    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v
    
    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        if v < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class IngestFileRequest(BaseModel):
    """Request to ingest a file."""
    
    plugin_name: str = Field(..., description="Name of the ingestion plugin to use")
    plugin_params: Dict[str, Any] = Field({}, description="Parameters for the plugin")


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    
    source: str = Field(..., description="Source of the document")
    filename: str = Field(..., description="Name of the source file")
    extension: str = Field(..., description="Extension of the source file")
    file_size: int = Field(..., description="Size of the source file in bytes")
    chunking_strategy: str = Field(..., description="Strategy used for chunking")
    chunk_unit: str = Field(..., description="Unit used for chunking")
    chunk_size: int = Field(..., description="Size of each chunk")
    chunk_overlap: int = Field(..., description="Overlap between chunks")
    chunk_index: int = Field(..., description="Index of this chunk")
    chunk_count: int = Field(..., description="Total number of chunks")
    document_id: Optional[str] = Field(None, description="ID of the document in ChromaDB")
    ingestion_timestamp: Optional[str] = Field(None, description="Timestamp when the document was ingested")


class Document(BaseModel):
    """A document chunk with metadata."""
    
    text: str = Field(..., description="Text content of the document")
    metadata: DocumentMetadata = Field(..., description="Metadata for the document")


class IngestFileResponse(BaseModel):
    """Response with the results of file ingestion."""
    
    file_path: str = Field(..., description="Path to the ingested file")
    document_count: int = Field(..., description="Number of document chunks created")
    documents: List[Document] = Field(..., description="List of document chunks")


class AddDocumentsRequest(BaseModel):
    """Request to add documents to a collection."""
    
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to add")


class AddDocumentsResponse(BaseModel):
    """Response with the results of adding documents to a collection."""
    
    collection_id: int = Field(..., description="ID of the collection")
    collection_name: str = Field(..., description="Name of the collection")
    documents_added: int = Field(..., description="Number of documents added")
    success: bool = Field(..., description="Whether the operation was successful")
    status: Optional[str] = Field(None, description="Status of the ingestion process (processing, completed, failed)")
    file_path: Optional[str] = Field(None, description="Path to the ingested file")
    file_url: Optional[str] = Field(None, description="URL to access the file")
    original_filename: Optional[str] = Field(None, description="Original filename")
    plugin_name: Optional[str] = Field(None, description="Name of the ingestion plugin used")
    file_registry_id: Optional[int] = Field(None, description="ID of the file registry entry")
