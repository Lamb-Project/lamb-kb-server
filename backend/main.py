import os
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
from routers import system_router
from database.connection import init_databases
from dependencies import verify_token
from plugins.base import discover_plugins
from routers import collections_router, files_router
from schemas.ingestion import IngestionPluginInfo
from services.ingestion_service import IngestionService


try:
    load_dotenv()
    print("INFO: Environment variables loaded from .env file")
    print(f"INFO: EMBEDDINGS_VENDOR={os.getenv('EMBEDDINGS_VENDOR')}")
    print(f"INFO: EMBEDDINGS_MODEL={os.getenv('EMBEDDINGS_MODEL')}")
except ImportError:
    print("WARNING: python-dotenv not installed, environment variables must be set manually")


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

    print("Discovering ingestion plugins...")
    discover_plugins("plugins")
    print(f"Found {len(IngestionService.list_plugins())} ingestion plugins")

    IngestionService._ensure_dirs()


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router.router)
app.include_router(collections_router.router)
app.include_router(files_router.router)
IngestionService._ensure_dirs()

static_dir = IngestionService.STATIC_DIR
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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