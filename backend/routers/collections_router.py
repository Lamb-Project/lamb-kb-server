# Python Libraries
import json
import os
import uuid

# Third-Party Libraries
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

# Local Imports
# Database imports
from database.connection import get_chroma_client, get_db, SessionLocal
from database.models import FileStatus
from database.service import CollectionService

# Dependency imports
from dependencies import verify_token

# Schema imports
from schemas.collection import CollectionCreate, CollectionList, CollectionResponse, EmbeddingsModel
from schemas.ingestion import AddDocumentsRequest, AddDocumentsResponse, IngestBaseRequest, IngestURLRequest
from schemas.query import QueryRequest, QueryResponse

# Service imports
from backend.services.ingestion_service import IngestionService
from backend.services.query_service import QueryService
from services.collection_service import CollectionService

router = APIRouter(
    prefix="/collections",
    tags=["Collections"],
    dependencies=[Depends(verify_token)]
)

# Helper function to get and validate collection existence in both databases
def _get_and_validate_collection(db: Session, collection_id: int):
    """
    Retrieves a collection by ID from SQLite and validates its existence in ChromaDB.
    
    Args:
        db: SQLAlchemy Session
        collection_id: ID of the collection
        
    Returns:
        Tuple: (collection_object, collection_name)
        
    Raises:
        HTTPException: If collection not found in either database.
    """
    collection = CollectionService.get_collection(db, collection_id)
    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database"
        )
    
    collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
    try:
        chroma_client = get_chroma_client()
        chroma_client.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
        )
        
    return collection, collection_name

@router.post(
    "/",
    response_model=CollectionResponse,
    summary="Create collection",
    description="Create a new knowledge base collection.",
    status_code=status.HTTP_201_CREATED
)
async def create_collection(
    collection: CollectionCreate,
    db: Session = Depends(get_db)
):
    """Create a new knowledge base collection."""
    if not collection.embeddings_model:
        return CollectionService.create_collection(collection, db)

    model_info = collection.embeddings_model.model_dump()
    resolved_config = {}

    vendor = model_info.get("vendor", "default")
    if vendor == "default":
        vendor = os.getenv("EMBEDDINGS_VENDOR")
    resolved_config["vendor"] = vendor

    model = model_info.get("model", "default")
    if model == "default":
        model = os.getenv("EMBEDDINGS_MODEL")
    resolved_config["model"] = model

    api_key = model_info.get("apikey", "default")
    if api_key == "default":
        api_key = os.getenv("EMBEDDINGS_APIKEY", "")
    resolved_config["apikey"] = api_key

    api_endpoint = model_info.get("api_endpoint")
    if not api_endpoint or api_endpoint == "default":
        api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT")

    if vendor == "ollama" and not api_endpoint:
        raise HTTPException(
            status_code=400,
            detail="Configuration error: 'ollama' vendor requires an API endpoint."
        )

    if api_endpoint:
        resolved_config["api_endpoint"] = api_endpoint

    collection.embeddings_model = EmbeddingsModel(**resolved_config)
    return CollectionService.create_collection(collection, db)

@router.get(
    "/",
    response_model=CollectionList,
    summary="List collections",
    description="List all available knowledge base collections."
)
async def list_collections(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    owner: str = Query(None),
    visibility: str = Query(None)
):
    """List all available knowledge base collections with optional filtering."""
    return CollectionService.list_collections(db, skip, limit, owner, visibility)

@router.get(
    "/{collection_id}",
    response_model=CollectionResponse,
    summary="Get collection",
    description="Get details of a specific knowledge base collection."
)
async def get_collection(collection_id: int, db: Session = Depends(get_db)):
    """Get details of a specific knowledge base collection."""
    return CollectionService.get_collection(collection_id, db)

@router.post(
    "/{collection_id}/ingest-url",
    response_model=AddDocumentsResponse,
    summary="Ingest content from URLs",
    description="Fetch, process, and add content from URLs to a collection.",
    tags=["Ingestion"]
)
async def ingest_url_to_collection(
    collection_id: int,
    request: IngestURLRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Ingest content from URLs directly into a collection."""
    collection, _ = _get_and_validate_collection(db, collection_id)
    
    plugin = IngestionService.get_plugin(request.plugin_name)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin {request.plugin_name} not found")

    
    collection_dir = IngestionService._get_collection_dir(collection.owner, collection.name)
    unique_filename = f"{uuid.uuid4().hex}.md"
    file_path = collection_dir / unique_filename
    
    first_url = request.urls[0] if request.urls else "unknown_url"
    file_registry = IngestionService.register_file(
        db=db,
        collection_id=collection_id,
        file_path=str(file_path),
        file_url=first_url,
        original_filename=first_url,
        plugin_name="url_ingest",
        plugin_params={"urls": request.urls, **request.plugin_params},
        owner=collection.owner,
        content_type="text/markdown",
        status=FileStatus.PROCESSING
    )
    
    def background_task(urls, plugin_name, params, coll_id, reg_id, f_path):
        db_bg = SessionLocal()
        try:
            plugin_instance = IngestionService.get_plugin(plugin_name)
            documents = plugin_instance.ingest(file_path=f_path, urls=urls, **params)
            IngestionService.add_documents_to_collection(db_bg, coll_id, documents)
            IngestionService.update_file_status(db_bg, reg_id, FileStatus.COMPLETED, len(documents))
        except Exception as e:
            print(f"ERROR: Background URL ingestion failed: {e}")
            IngestionService.update_file_status(db_bg, reg_id, FileStatus.FAILED)
        finally:
            db_bg.close()

    background_tasks.add_task(
        background_task,
        request.urls,
        request.plugin_name,
        request.plugin_params,
        collection_id,
        file_registry.id,
        str(file_path)
    )
    
    return {
        "collection_id": collection_id,
        "collection_name": collection.name,
        "documents_added": 0,
        "success": True,
        "file_path": str(file_path),
        "file_url": "",
        "original_filename": f"urls_{len(request.urls)}",
        "plugin_name": request.plugin_name,
        "file_registry_id": file_registry.id,
        "status": "processing"
    }


@router.post(
    "/{collection_id}/documents",
    response_model=AddDocumentsResponse,
    summary="Add documents to a collection",
    description="Add processed documents to a collection.",
    tags=["Ingestion"]
)
async def add_documents(
    collection_id: int,
    request: AddDocumentsRequest,
    db: Session = Depends(get_db)
):
    """Add documents to a collection."""
    _get_and_validate_collection(db, collection_id)
    return IngestionService.add_documents_to_collection(db, collection_id, request.documents)


@router.post(
    "/{collection_id}/ingest-file",
    response_model=AddDocumentsResponse,
    summary="Ingest a file into a collection",
    description="Upload, process, and add a file to a collection.",
    tags=["Ingestion"]
)
async def ingest_file_to_collection(
    collection_id: int,
    file: UploadFile = File(...),
    plugin_name: str = Form(...),
    plugin_params: str = Form("{}"),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Ingest a file directly into a collection."""
    collection, collection_name = _get_and_validate_collection(db, collection_id)
    
    if not IngestionService.get_plugin(plugin_name):
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_name}' not found")
    
    try:
        params = json.loads(plugin_params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in plugin_params")
    
    file_info = IngestionService.save_uploaded_file(file, collection.owner, collection_name)
    
    file_registry = IngestionService.register_file(
        db=db,
        collection_id=collection_id,
        file_path=file_info["file_path"],
        file_url=file_info["file_url"],
        original_filename=file_info["original_filename"],
        plugin_name=plugin_name,
        plugin_params=params,
        owner=collection.owner,
        content_type=file.content_type,
        status=FileStatus.PROCESSING
    )
    
    def background_task(f_path, p_name, p_params, coll_id, reg_id):
        
        db_bg = SessionLocal()
        try:
            documents = IngestionService.ingest_file(f_path, p_name, p_params)
            IngestionService.add_documents_to_collection(db_bg, coll_id, documents)
            IngestionService.update_file_status(db_bg, reg_id, FileStatus.COMPLETED, len(documents))
        except Exception as e:
            print(f"ERROR: Background file ingestion failed: {e}")
            IngestionService.update_file_status(db_bg, reg_id, FileStatus.FAILED)
        finally:
            db_bg.close()

    background_tasks.add_task(
        background_task,
        file_info["file_path"],
        plugin_name,
        params,
        collection_id,
        file_registry.id
    )
    
    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "documents_added": 0,
        "success": True,
        **file_info,
        "plugin_name": plugin_name,
        "file_registry_id": file_registry.id,
        "status": "processing"
    }

@router.post(
    "/{collection_id}/query",
    response_model=QueryResponse,
    summary="Query a collection",
    description="Query a collection using a specified plugin.",
    tags=["Query"]
)
async def query_collection(
    collection_id: int,
    request: QueryRequest,
    plugin_name: str = Query("simple_query"),
    db: Session = Depends(get_db)
):
    """Query a collection using a specified plugin."""
    _get_and_validate_collection(db, collection_id)
    
    plugin_params = request.plugin_params or {}
    if "top_k" not in plugin_params and request.top_k is not None:
        plugin_params["top_k"] = request.top_k
    if "threshold" not in plugin_params and request.threshold is not None:
        plugin_params["threshold"] = request.threshold
        
    return QueryService.query_collection(
        db, collection_id, request.query_text, plugin_name, plugin_params
    )

@router.post(
    "/{collection_id}/ingest-base",
    response_model=AddDocumentsResponse,
    summary="Ingest content using a base-ingest plugin",
    description="Process and add content to a collection using a base-ingest plugin.",
    tags=["Ingestion"]
)
async def ingest_base_to_collection(
    collection_id: int,
    request: IngestBaseRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Ingest content using a base-ingest plugin."""
    collection, collection_name = _get_and_validate_collection(db, collection_id)
    
    plugin = IngestionService.get_plugin(request.plugin_name)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin {request.plugin_name} not found")
    if plugin.kind != "base-ingest":
        raise HTTPException(status_code=400, detail=f"Plugin {request.plugin_name} is not a base-ingest plugin")
    
    collection_dir = IngestionService._get_collection_dir(collection.owner, collection_name)
    unique_filename = f"{uuid.uuid4().hex}.md"
    file_path = collection_dir / unique_filename
    
    file_registry = IngestionService.register_file(
        db=db,
        collection_id=collection_id,
        file_path=str(file_path),
        original_filename=f"{request.plugin_name}_{unique_filename}",
        plugin_name=request.plugin_name,
        plugin_params=request.plugin_params,
        owner=collection.owner,
        content_type="text/markdown",
        status=FileStatus.PROCESSING
    )
    
    def background_task(p_name, params, coll_id, reg_id, f_path):
        db_bg = SessionLocal()
        try:
            plugin_instance = IngestionService.get_plugin(p_name)
            documents = plugin_instance.ingest(file_path=f_path, **params)
            IngestionService.add_documents_to_collection(db_bg, coll_id, documents)
            IngestionService.update_file_status(db_bg, reg_id, FileStatus.COMPLETED, len(documents))
        except Exception as e:
            print(f"ERROR: Background base ingestion failed: {e}")
            IngestionService.update_file_status(db_bg, reg_id, FileStatus.FAILED)
        finally:
            db_bg.close()

    background_tasks.add_task(
        background_task,
        request.plugin_name,
        request.plugin_params,
        collection_id,
        file_registry.id,
        str(file_path)
    )
    
    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "documents_added": 0,
        "success": True,
        "file_path": str(file_path),
        "file_url": "",
        "original_filename": f"{request.plugin_name}_{unique_filename}",
        "plugin_name": request.plugin_name,
        "file_registry_id": file_registry.id,
        "status": "processing"
    }
