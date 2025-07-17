"""
Collection service module for handling collection-related business logic.
"""

# Python Libraries
import json
from typing import Any, Dict, Optional

# Third-Party Libraries
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

# Local Imports
# Database imports
from database.connection import get_embedding_function
from database.models import Collection, Visibility
from database.service import CollectionService as DBCollectionService

# Schema imports
from schemas.collection import CollectionCreate

class CollectionService:
    """Service for handling collection-related operations."""
    
    @staticmethod
    def create_collection(
        collection: CollectionCreate,
        db: Session,
    ) -> Dict[str, Any]:
        """Create a new knowledge base collection."""
        existing = DBCollectionService.get_collection_by_name(db, collection.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection with name '{collection.name}' already exists"
            )
        
        try:
            visibility = Visibility(collection.visibility)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid visibility value: {collection.visibility}. Must be 'private' or 'public'."
            )
        
        try:
            embeddings_model = {}
            if collection.embeddings_model:
                model_info = collection.embeddings_model.model_dump()
                temp_collection = Collection(id=-1, name="temp_validation", 
                                            owner="system", description="Validation only", 
                                            embeddings_model=json.dumps(model_info))
                try:
                    embedding_function = get_embedding_function(temp_collection)
                    embedding_function(["Test embedding validation"])
                except Exception as emb_error:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Embeddings model validation failed: {str(emb_error)}"
                    )
                embeddings_model = model_info
            
            db_collection = DBCollectionService.create_collection(
                db=db,
                name=collection.name,
                owner=collection.owner,
                description=collection.description,
                visibility=visibility,
                embeddings_model=embeddings_model
            )
            
            if isinstance(db_collection.embeddings_model, str):
                db_collection.embeddings_model = json.loads(db_collection.embeddings_model or '{}')
            
            if not db_collection.chromadb_uuid:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Collection created but ChromaDB UUID was not stored"
                )
            
            return db_collection
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create collection: {str(e)}"
            )
    
    @staticmethod
    def list_collections(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        owner: Optional[str] = None,
        visibility: Optional[str] = None
    ) -> Dict[str, Any]:
        """List all available knowledge base collections with optional filtering."""
        visibility_enum = None
        if visibility:
            try:
                visibility_enum = Visibility(visibility)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid visibility value: {visibility}."
                )
        
        collections = DBCollectionService.list_collections(
            db=db, owner=owner, visibility=visibility_enum, skip=skip, limit=limit
        )
        
        query = db.query(Collection)
        if owner:
            query = query.filter(Collection.owner == owner)
        if visibility_enum:
            query = query.filter(Collection.visibility == visibility_enum)
        total = query.count()
        
        return {"total": total, "items": collections}
    
    @staticmethod
    def get_collection(
        collection_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """Get details of a specific knowledge base collection."""
        collection = DBCollectionService.get_collection(db, collection_id)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection with ID {collection_id} not found"
            )
        return collection
