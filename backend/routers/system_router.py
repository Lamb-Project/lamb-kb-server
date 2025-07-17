# Python Libraries
from typing import Any, Dict

# Third-Party Libraries
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

# Local Imports
# Database imports
from database.connection import Collection, get_chroma_client, get_db, init_databases
from database.models import Collection

# Dependency imports
from dependencies import verify_token

# Schema imports
from schemas.system import DatabaseStatusResponse, HealthResponse, MessageResponse

router = APIRouter()


# Root endpoint with enhanced documentation
@router.get(
    "/", 
    response_model=MessageResponse,
    summary="Root endpoint",
    description="""Returns a welcome message to confirm the server is running.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["System"],
    responses={
        200: {"description": "Successful response with welcome message"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def root(token: str = Depends(verify_token)):
    """Root endpoint that returns a welcome message.
    
    This endpoint is primarily used to verify the server is running and authentication is working correctly.
    
    Returns:
        A dictionary containing a welcome message
    """
    return {"message": "Hello World from the Lamb Knowledge Base Server!"}


# Health check endpoint
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="""Check the health status of the server.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/health'
    ```
    """,
    tags=["System"],
    responses={
        200: {"description": "Server is healthy and running"}
    }
)
async def health_check():
    """Health check endpoint that does not require authentication.
    
    This endpoint can be used to verify the server is running without requiring authentication.
    
    Returns:
        A dictionary containing the server status and version
    """
    return {"status": "ok", "version": "0.1.0"}


# Database status endpoint
@router.get(
    "/database/status",
    response_model=DatabaseStatusResponse,
    summary="Database status",
    description="""Check the status of all databases.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/database/status' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Database"],
    responses={
        200: {"description": "Database status information"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def database_status(token: str = Depends(verify_token), db: Session = Depends(get_db)):
    """Get the status of the SQLite and ChromaDB databases.
    
    Returns:
        A dictionary with database status information
    """
    # Re-initialize databases to get fresh status
    db_status = init_databases()
    
    # Count collections in SQLite
    collections_count = db.query(Collection).count()
    
    # Get ChromaDB collections
    chroma_client = get_chroma_client()
    chroma_collections = chroma_client.list_collections()
    
    return {
        "sqlite_status": {
            "initialized": db_status["sqlite_initialized"],
            "schema_valid": db_status["sqlite_schema_valid"],
            "errors": db_status.get("errors", [])
        },
        "chromadb_status": {
            "initialized": db_status["chromadb_initialized"],
            "collections_count": len(chroma_collections)
        },
        "collections_count": collections_count
    } 