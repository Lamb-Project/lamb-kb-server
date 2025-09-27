"""
Collections service module for handling collection-related endpoint logic.

This module provides service functions for handling collection-related API endpoints,
separating the business logic from the FastAPI route definitions.
"""

import json
import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database.models import Collection, Visibility, FileRegistry, FileStatus
from database.service import CollectionService as DBCollectionService
from services.ingestion import IngestionService
from schemas.collection import (
    CollectionCreate, 
    CollectionUpdate, 
    CollectionResponse, 
    CollectionList
)
from database.connection import get_embedding_function
from database.connection import get_chroma_client

class CollectionsService:
    """Service for handling collection-related API endpoints."""
    
    @staticmethod
    def create_collection(
        collection: CollectionCreate,
        db: Session,
    ) -> Dict[str, Any]:
        """Create a new knowledge base collection.
        
        Args:
            collection: Collection data from request body with resolved default values
            db: Database session
            
        Returns:
            The created collection
            
        Raises:
            HTTPException: If collection creation fails
        """
        # Check if collection with this name already exists
        existing = DBCollectionService.get_collection_by_name(db, collection.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection with name '{collection.name}' already exists"
            )
        
        # Convert visibility string to enum
        try:
            visibility = Visibility(collection.visibility)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid visibility value: {collection.visibility}. Must be 'private' or 'public'."
            )
        
        # Create the collection
        try:
            # Handle the embeddings model configuration
            embeddings_model = {}
            if collection.embeddings_model:
                # Get the model values from the request
                # Note: Default values should already be resolved by main.py
                model_info = collection.embeddings_model.model_dump()
                
                # We'll still validate the embeddings model configuration
                try:
                    # Create a temporary DB collection record for validation
                    from database.models import Collection
                    temp_collection = Collection(id=-1, name="temp_validation", 
                                                owner="system", description="Validation only", 
                                                embeddings_model=json.dumps(model_info))
                    
                    # No logging of API key details, only log the vendor and model
                    if model_info.get('vendor', '').lower() == 'openai':
                        print(f"DEBUG: [create_collection] Validating OpenAI embeddings with model: {model_info.get('model')}")
                    
                    # Try to create an embedding function with this configuration
                    # This will validate if the embeddings model configuration is valid
                    embedding_function = get_embedding_function(temp_collection)
                    
                    # Test the embedding function with a simple text
                    test_result = embedding_function(["Test embedding validation"])
                    print(f"INFO: Embeddings validation successful, dimensions: {len(test_result[0])}")
                except Exception as emb_error:
                    print(f"ERROR: Embeddings model validation failed: {str(emb_error)}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Embeddings model validation failed: {str(emb_error)}. Please check your configuration."
                    )
                
                embeddings_model = model_info
            
            # Create the collection in both databases
            db_collection = DBCollectionService.create_collection(
                db=db,
                name=collection.name,
                owner=collection.owner,
                description=collection.description,
                visibility=visibility,
                embeddings_model=embeddings_model
            )
            
            # Ensure embeddings_model is a dictionary before returning
            if isinstance(db_collection.embeddings_model, str):
                try:
                    db_collection.embeddings_model = json.loads(db_collection.embeddings_model)
                except (json.JSONDecodeError, TypeError):
                    # If we can't parse it, return an empty dict rather than failing
                    db_collection.embeddings_model = {}
            
            # Verify the collection was created successfully in both databases
            if not db_collection.chromadb_uuid:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Collection was created but ChromaDB UUID was not stored"
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
        """List all available knowledge base collections with optional filtering.
        
        Args:
            db: Database session
            skip: Number of collections to skip
            limit: Maximum number of collections to return
            owner: Optional filter by owner
            visibility: Optional filter by visibility
            
        Returns:
            Dict with total count and list of collections
            
        Raises:
            HTTPException: If invalid visibility value is provided
        """
        # Convert visibility string to enum if provided
        visibility_enum = None
        if visibility:
            try:
                visibility_enum = Visibility(visibility)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid visibility value: {visibility}. Must be 'private' or 'public'."
                )
        
        # Get collections with filtering
        collections = DBCollectionService.list_collections(
            db=db,
            owner=owner,
            visibility=visibility_enum,
            skip=skip,
            limit=limit
        )
        
        # Count total collections with same filter
        query = db.query(Collection)
        if owner:
            query = query.filter(Collection.owner == owner)
        if visibility_enum:
            query = query.filter(Collection.visibility == visibility_enum)
        total = query.count()
        
        return {
            "total": total,
            "items": collections
        }
    
    @staticmethod
    def get_collection(
        collection_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """Get details of a specific knowledge base collection.
        
        Args:
            collection_id: ID of the collection to retrieve
            db: Database session
            
        Returns:
            Collection details
            
        Raises:
            HTTPException: If collection not found
        """
        collection = DBCollectionService.get_collection(db, collection_id)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection with ID {collection_id} not found"
            )
        return collection
    
    @staticmethod
    def list_files(
        collection_id: int,
        db: Session,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all files in a collection.
        
        Args:
            collection_id: ID of the collection
            db: Database session
            status: Optional filter by status
            
        Returns:
            List of file registry entries
            
        Raises:
            HTTPException: If collection not found or status invalid
        """
        # Check if collection exists
        collection = DBCollectionService.get_collection(db, collection_id)
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection with ID {collection_id} not found"
            )
        
        # Query files
        query = db.query(FileRegistry).filter(FileRegistry.collection_id == collection_id)
        
        # Apply status filter if provided
        if status:
            try:
                file_status = FileStatus(status)
                query = query.filter(FileRegistry.status == file_status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
                )
        
        # Get results
        files = query.all()
        
        # Convert to response model
        return [file.to_dict() for file in files]
    
    @staticmethod
    def update_file_status(
        file_id: int,
        status: str,
        db: Session
    ) -> Dict[str, Any]:
        """Update the status of a file in the registry.
        
        Args:
            file_id: ID of the file registry entry
            status: New status
            db: Database session
            
        Returns:
            Updated file registry entry
            
        Raises:
            HTTPException: If file not found or status invalid
        """
        # Validate status
        try:
            file_status = FileStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: completed, processing, failed, deleted"
            )
        
        # Update file status
        file = IngestionService.update_file_status(db, file_id, file_status)
        if not file:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found"
            )
        
        return file.to_dict()
        
    @staticmethod
    def get_file_content(file_id: int, db: Session) -> Dict[str, Any]:
        """Get the content of a file.
        
        Args:
            file_id: ID of the file registry entry
            db: Database session
            
        Returns:
            Content of the file with metadata
            
        Raises:
            HTTPException: If file not found or content cannot be retrieved
        """
        import mimetypes
        from pathlib import Path
        
        # Get file registry entry
        file_registry = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if not file_registry:
            raise HTTPException(
                status_code=404,
                detail=f"File with ID {file_id} not found"
            )
        
        # Get collection
        collection = db.query(Collection).filter(Collection.id == file_registry.collection_id).first()
        if not collection:
            raise HTTPException(
                status_code=404,
                detail=f"Collection with ID {file_registry.collection_id} not found"
            )
            
        # Get ChromaDB client and collection
        chroma_client = get_chroma_client()
        
        # Get the embedding function for this collection
        from database.connection import get_embedding_function
        embedding_function = get_embedding_function(collection)
        
        chroma_collection = chroma_client.get_collection(
            name=collection.name,
            embedding_function=embedding_function
        )
        
        # Debug: Let's see what metadata we have in ChromaDB
        print(f"DEBUG: Looking for file with ID {file_id}")
        print(f"DEBUG: File registry - original_filename: {file_registry.original_filename}")
        print(f"DEBUG: File registry - file_path: {file_registry.file_path}")
        
        # First, let's get ALL documents to see their metadata structure
        all_results = chroma_collection.get(
            include=["metadatas"],
            limit=1  # Just get one to see the structure
        )
        
        if all_results["metadatas"]:
            print(f"DEBUG: Sample metadata from ChromaDB: {all_results['metadatas'][0]}")
        
        # The metadata uses "source" field but the value should match file_path from ingestion
        # Let's try multiple query strategies
        results = None
        
        # Strategy 1: Query by source field with file_path
        results = chroma_collection.get(
            where={"source": file_registry.file_path},
            include=["documents", "metadatas"]
        )
        
        print(f"DEBUG: Query by source={file_registry.file_path} returned {len(results['documents'])} documents")
        
        # Strategy 2: If no results, try by filename field
        if not results["documents"]:
            results = chroma_collection.get(
                where={"filename": file_registry.original_filename},
                include=["documents", "metadatas"]
            )
            print(f"DEBUG: Query by filename={file_registry.original_filename} returned {len(results['documents'])} documents")
        
        # Strategy 3: If still no results, get all and filter manually
        if not results["documents"]:
            print("DEBUG: Trying to get all documents and filter manually")
            all_results = chroma_collection.get(
                include=["documents", "metadatas"]
            )
            
            # Filter by matching source or filename
            filtered_docs = []
            filtered_metas = []
            
            for i, meta in enumerate(all_results["metadatas"]):
                if meta:
                    # Check if source matches either file_path or contains original_filename
                    source_value = meta.get("source", "")
                    filename_value = meta.get("filename", "")
                    
                    print(f"DEBUG: Checking document {i}: source={source_value}, filename={filename_value}")
                    
                    if (source_value == file_registry.file_path or 
                        source_value.endswith(file_registry.original_filename) or
                        filename_value == file_registry.original_filename):
                        filtered_docs.append(all_results["documents"][i])
                        filtered_metas.append(meta)
            
            if filtered_docs:
                results = {
                    "documents": filtered_docs,
                    "metadatas": filtered_metas
                }
                print(f"DEBUG: Manual filtering found {len(filtered_docs)} documents")
        
        # If no content in ChromaDB, raise error
        if not results or not results["documents"]:
            raise HTTPException(
                status_code=404,
                detail=f"No content found for file: {file_registry.original_filename}"
            )
        
        # Reconstruct content from chunks
        chunk_docs = []
        for i, doc in enumerate(results["documents"]):
            if i < len(results["metadatas"]) and results["metadatas"][i]:
                metadata = results["metadatas"][i]
                chunk_docs.append({
                    "text": doc,
                    "index": metadata.get("chunk_index", i),
                    "count": metadata.get("chunk_count", 0)
                })
        
        # Sort chunks by index
        chunk_docs.sort(key=lambda x: x["index"])
        
        # Join all chunks
        full_content = "\n".join(doc["text"] for doc in chunk_docs)
        
        # Detect content type based on file extension
        content_type = "text/plain"  # Default
        
        # Check if it's a URL ingestion (original_filename starts with http)
        if file_registry.original_filename.startswith(('http://', 'https://')):
            content_type = "text/html"
        else:
            # Get file extension from original filename
            file_path = Path(file_registry.original_filename)
            file_extension = file_path.suffix.lower()
            
            # Map extensions to content types
            content_type_mapping = {
                '.md': 'text/markdown',
                '.markdown': 'text/markdown',
                '.txt': 'text/plain',
                '.text': 'text/plain',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.xml': 'text/xml',
                '.json': 'application/json',
                '.yaml': 'text/yaml',
                '.yml': 'text/yaml',
                '.csv': 'text/csv',
                '.tsv': 'text/tab-separated-values',
                '.rst': 'text/x-rst',
                '.tex': 'text/x-tex',
                '.py': 'text/x-python',
                '.js': 'text/javascript',
                '.css': 'text/css',
                '.sh': 'text/x-shellscript',
                '.bat': 'text/x-batch',
                '.ps1': 'text/x-powershell',
                '.sql': 'text/x-sql',
                '.log': 'text/plain',
                '.ini': 'text/plain',
                '.cfg': 'text/plain',
                '.conf': 'text/plain',
                '.properties': 'text/plain',
                '.toml': 'text/plain',
            }
            
            # Check our mapping first
            if file_extension in content_type_mapping:
                content_type = content_type_mapping[file_extension]
            else:
                # Fall back to Python's mimetypes module
                guessed_type, _ = mimetypes.guess_type(file_registry.original_filename)
                if guessed_type and guessed_type.startswith('text'):
                    content_type = guessed_type
                # For special cases where mimetypes might not recognize
                elif file_extension in ['.ipynb']:
                    content_type = 'application/json'
        
        # Check if there's a .html version of the file (for markitdown plugin)
        # The markitdown plugin creates .html files from various formats
        if file_extension in ['.pdf', '.pptx', '.docx', '.xlsx', '.xls', '.epub', '.zip']:
            # These files are likely processed by markitdown plugin which creates HTML
            content_type = 'text/html'
            
            # Also check if the content looks like HTML
            if full_content.strip().startswith(('<html', '<!DOCTYPE', '<HTML', '<!doctype')):
                content_type = 'text/html'
        
        print(f"DEBUG: Detected content type: {content_type} for file: {file_registry.original_filename}")
        
        return {
            "file_id": file_id,
            "original_filename": file_registry.original_filename,
            "content": full_content,
            "content_type": content_type,
            "chunk_count": len(chunk_docs),
            "timestamp": file_registry.updated_at.isoformat() if file_registry.updated_at else None
        }