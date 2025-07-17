"""
File service module for handling file-related business logic.
"""

# Python Libraries
from typing import Any, Dict, List, Optional

# Third-Party Libraries
from fastapi import HTTPException
from sqlalchemy.orm import Session

# Local Imports
# Database imports
from database.connection import get_chroma_client
from database.models import Collection, FileRegistry, FileStatus
from database.service import CollectionService as DBCollectionService

# Service imports
from backend.services.ingestion_service import IngestionService

class FileService:
    """Service for handling file-related operations."""

    @staticmethod
    def list_files(
        collection_id: int,
        db: Session,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all files in a collection."""
        collection = DBCollectionService.get_collection(db, collection_id)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection with ID {collection_id} not found"
            )
        
        query = db.query(FileRegistry).filter(FileRegistry.collection_id == collection_id)
        
        if status:
            try:
                file_status = FileStatus(status)
                query = query.filter(FileRegistry.status == file_status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}."
                )
        
        files = query.all()
        return [file.to_dict() for file in files]
    
    @staticmethod
    def update_file_status(
        file_id: int,
        status: str,
        db: Session
    ) -> Dict[str, Any]:
        """Update the status of a file in the registry."""
        try:
            file_status = FileStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}."
            )
        
        file = IngestionService.update_file_status(db, file_id, file_status)
        if not file:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found"
            )
        
        return file.to_dict()
        
    @staticmethod
    def get_file_content(file_id: int, db: Session) -> Dict[str, Any]:
        """Get the content of a file."""
        file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if not file_registry:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found"
            )
        
        collection = db.query(Collection).filter(Collection.id == file_registry.collection_id).first()
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection for file ID {file_id} not found"
            )
            
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection.name)
        
        source = file_registry.original_filename
        
        results = chroma_collection.get(
            where={"filename": source}, 
            include=["documents", "metadatas"]
        )

        if not results["documents"]:
            raise HTTPException(
                status_code=404,
                detail=f"No content found for file: {source}"
            )
        
        chunk_docs = []
        for i, doc in enumerate(results["documents"]):
            if i < len(results["metadatas"]) and results["metadatas"][i]:
                metadata = results["metadatas"][i]
                chunk_docs.append({
                    "text": doc,
                    "index": metadata.get("chunk_index", i),
                    "count": metadata.get("chunk_count", 0)
                })
        
        chunk_docs.sort(key=lambda x: x["index"])
        full_content = "\\n".join(doc["text"] for doc in chunk_docs)
        
        return {
            "file_id": file_id,
            "original_filename": source,
            "content": full_content,
            "content_type": "markdown",
            "chunk_count": len(chunk_docs),
            "timestamp": file_registry.updated_at.isoformat() if file_registry.updated_at else None
        }
