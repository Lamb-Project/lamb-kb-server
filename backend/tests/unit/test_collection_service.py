from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from schemas.collection import CollectionCreate, EmbeddingsModel
from services.collection_service import CollectionService

# Mock data for a collection returned from the database
mock_collection_data = {
    "id": 1,
    "name": "test_collection",
    "owner": "test_user",
    "description": "A test collection",
    "visibility": "private",
    "creation_date": datetime.now(),
    "embeddings_model": {
        "model": "nomic-embed-text",
        "vendor": "ollama",
    },
    "chromadb_uuid": "some-uuid-1234"
}

# Mock for the database session
db = MagicMock(spec=Session)

# Test case for creating a collection successfully
def test_create_collection_success():
    # Mock the DB and embedding services to avoid actual DB/network operations
    with patch('services.collection_service.DBCollectionService.get_collection_by_name', return_value=None), \
         patch('services.collection_service.get_embedding_function', return_value=MagicMock()), \
         patch('services.collection_service.DBCollectionService.create_collection', return_value=SimpleNamespace(**mock_collection_data)):

        collection_create = CollectionCreate(
            name="test_collection",
            owner="test_user",
            description="A test collection",
            visibility="private",
            embeddings_model=EmbeddingsModel(
                model="nomic-embed-text",
                vendor="ollama"
            )
        )
        result = CollectionService.create_collection(collection_create, db)
        assert result.name == "test_collection"

# Test case for attempting to create a collection that already exists
def test_create_collection_already_exists():
    with patch('services.collection_service.DBCollectionService.get_collection_by_name', return_value=SimpleNamespace(**mock_collection_data)):
        collection_create = CollectionCreate(
            name="test_collection",
            owner="test_user",
            description="A test collection",
            visibility="private",
            embeddings_model=EmbeddingsModel(
                model="nomic-embed-text",
                vendor="ollama"
            )
        )
        with pytest.raises(HTTPException) as excinfo:
            CollectionService.create_collection(collection_create, db)
        assert excinfo.value.status_code == 409

# Test case for creating a collection with invalid visibility
def test_create_collection_invalid_visibility():
    with patch('services.collection_service.DBCollectionService.get_collection_by_name', return_value=None):
        collection_create = CollectionCreate(
            name="test_collection",
            owner="test_user",
            description="A test collection",
            visibility="invalid_visibility",
            embeddings_model=EmbeddingsModel(
                model="nomic-embed-text",
                vendor="ollama"
            )
        )
        with pytest.raises(HTTPException) as excinfo:
            CollectionService.create_collection(collection_create, db)
        assert excinfo.value.status_code == 400

# Test case for successfully listing collections
def test_list_collections_success():
    # Mock the DB service to return a list with a single item and mock the count
    with patch('services.collection_service.DBCollectionService.list_collections', return_value=[SimpleNamespace(**mock_collection_data)]), \
         patch.object(db, 'query') as mock_query:
        mock_query.return_value.count.return_value = 1
        result = CollectionService.list_collections(db)
        assert len(result['items']) == 1
        assert result['total'] == 1
        assert result['items'][0].name == "test_collection"

# Test case for listing collections when the database is empty
def test_list_collections_empty():
    # Mock the DB service to return an empty list and mock the count
    with patch('services.collection_service.DBCollectionService.list_collections', return_value=[]), \
         patch.object(db, 'query') as mock_query:
        mock_query.return_value.count.return_value = 0
        result = CollectionService.list_collections(db)
        assert len(result['items']) == 0
        assert result['total'] == 0

# Test case for getting a single collection successfully
def test_get_collection_success():
    with patch('services.collection_service.DBCollectionService.get_collection', return_value=SimpleNamespace(**mock_collection_data)):
        result = CollectionService.get_collection(1, db)
        assert result.name == "test_collection"

# Test case for attempting to get a non-existent collection
def test_get_collection_not_found():
    with patch('services.collection_service.DBCollectionService.get_collection', return_value=None):
        with pytest.raises(HTTPException) as excinfo:
            CollectionService.get_collection(999, db)
        assert excinfo.value.status_code == 404

# Test case for creating a collection with empty embeddings
def test_create_collection_with_empty_embeddings():
    with patch('services.collection_service.DBCollectionService.get_collection_by_name', return_value=None), \
         patch('services.collection_service.get_embedding_function', return_value=MagicMock()), \
         patch('services.collection_service.DBCollectionService.create_collection', return_value=SimpleNamespace(**mock_collection_data)):
        collection_create = CollectionCreate(
            name="test_collection_no_embeddings",
            owner="test_user",
            description="A test collection",
            visibility="private",
            embeddings_model=None
        )
        result = CollectionService.create_collection(collection_create, db)
        assert result.name == "test_collection"