import os
import json
from typing import Dict, Any, List, Optional
from backend.routers import system_router
from routers import collections_router, files_router

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load the environment variables from .env file
    load_dotenv()
    print(f"INFO: Environment variables loaded from .env file")
    print(f"INFO: EMBEDDINGS_VENDOR={os.getenv('EMBEDDINGS_VENDOR')}")
    print(f"INFO: EMBEDDINGS_MODEL={os.getenv('EMBEDDINGS_MODEL')}")
except ImportError:
    print("WARNING: python-dotenv not installed, environment variables must be set manually")

from fastapi import Depends, FastAPI, HTTPException, status, Query, File, Form, UploadFile, BackgroundTasks
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Database imports
from database.connection import init_databases, get_db, get_chroma_client
from database.models import Visibility, FileRegistry, FileStatus
from database.service import CollectionService
from schemas.collection import (
    CollectionUpdate,
    EmbeddingsModel
)

# Import ingestion modules
from plugins.base import discover_plugins
from backend.services.ingestion_service import IngestionService
from backend.services.query_service import QueryService
from schemas.ingestion import (
    IngestionPluginInfo,
    IngestFileResponse,
    IngestURLResponse,
)

# Import query modules
from schemas.query import (
    QueryPluginInfo
)

# Get API key from environment variable or use default
API_KEY = os.getenv("LAMB_API_KEY", "0p3n-w3bu!")

# Get default embeddings model configuration from environment variables
# Default to using Ollama with nomic-embed-text model
# For OpenAI models, the environment variables should be set accordingly
DEFAULT_EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
DEFAULT_EMBEDDINGS_VENDOR = os.getenv("EMBEDDINGS_VENDOR", "ollama")  # 'ollama', 'local', or 'openai'
DEFAULT_EMBEDDINGS_APIKEY = os.getenv("EMBEDDINGS_APIKEY", "")
# Default endpoint for Ollama
DEFAULT_EMBEDDINGS_ENDPOINT = os.getenv("EMBEDDINGS_ENDPOINT", "http://localhost:11434/api/embeddings")

# Initialize FastAPI app with detailed documentation
app = FastAPI(
    title="Lamb Knowledge Base Server",
    description="""A dedicated knowledge base server designed to provide robust vector database functionality 
    for the LAMB project and to serve as a Model Context Protocol (MCP) server.
    
    ## Authentication
    
    All API endpoints are secured with Bearer token authentication. The token must match 
    the `LAMB_API_KEY` environment variable (default: `0p3n-w3bu!`).
    
    Example:
    ```
    curl -H 'Authorization: Bearer 0p3n-w3bu!' http://localhost:9090/
    ```
    
    ## Features
    
    - Knowledge base management for LAMB Learning Assistants
    - Vector database services using ChromaDB
    - API access for the LAMB project
    - Model Context Protocol (MCP) compatibility
    """,
    version="0.1.0",
    contact={
        "name": "LAMB Project Team",
    },
    license_info={
        "name": "GNU General Public License v3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.en.html"
    },
)

# Import shared dependencies
from dependencies import verify_token

# Import routers
from routers import collections_router, files_router

# Initialize databases on startup
@app.on_event("startup")
async def startup_event():
    """Initialize databases and perform sanity checks on startup."""
    print("Initializing databases...")
    init_status = init_databases()
    
    if init_status["errors"]:
        for error in init_status["errors"]:
            print(f"ERROR: {error}")
    else:
        print("Databases initialized successfully.")
    
    # Discover ingestion plugins
    print("Discovering ingestion plugins...")
    discover_plugins("plugins")
    print(f"Found {len(IngestionService.list_plugins())} ingestion plugins")
    
    # Ensure static directory exists
    IngestionService._ensure_dirs()

# Include routers
app.include_router(system_router.router)
app.include_router(collections_router.router)
app.include_router(files_router.router)
IngestionService._ensure_dirs()

# Configure static files
static_dir = IngestionService.STATIC_DIR
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ingestion Plugin Endpoints

@app.get(
    "/ingestion/plugins",
    response_model=List[IngestionPluginInfo],
    summary="List ingestion plugins",
    description="""List all available document ingestion plugins.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/ingestion/plugins' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Ingestion"],
    responses={
        200: {"description": "List of available ingestion plugins"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def list_ingestion_plugins(token: str = Depends(verify_token)):
    """List all available document ingestion plugins.
    
    Returns:
        List of plugin information objects
    """
    return IngestionService.list_plugins()