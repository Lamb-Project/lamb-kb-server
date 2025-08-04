from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from starlette.testclient import TestClient

# Mock data to be returned by the service layer
mock_collection_data = {
    "id": 1,
    "name": "test_collection",
    "owner": "test_user",
    "description": "A test collection",
    "visibility": "private",
    "creation_date": datetime.now().isoformat(),
    "embeddings_model": {
        "model": "nomic-embed-text",
        "vendor": "ollama",
    },
    "chromadb_uuid": "some-uuid-1234"
}
mock_collection = SimpleNamespace(**mock_collection_data)

# Test case for successfully creating a collection
def test_create_collection(client: TestClient):
    with patch('services.collection_service.CollectionService.create_collection', return_value=mock_collection):
        response = client.post("/collections", json={
            "name": "test_collection",
            "owner": "test_user",
            "description": "A test collection",
            "visibility": "private",
            "embeddings_model": {
                "model": "nomic-embed-text",
                "vendor": "ollama"
            }
        })
        assert response.status_code == 201
        assert response.json()["name"] == "test_collection"

# Test case for listing all collections
def test_list_collections(client: TestClient):
    with patch('services.collection_service.CollectionService.list_collections', return_value={"total": 1, "items": [mock_collection_data]}):
        response = client.get("/collections")
        assert response.status_code == 200
        assert len(response.json()["items"]) == 1
        assert response.json()["items"][0]["name"] == "test_collection"

# Test case for listing collections when there are none
def test_list_collections_empty(client: TestClient):
    with patch('services.collection_service.CollectionService.list_collections', return_value={"total": 0, "items": []}):
        response = client.get("/collections")
        assert response.status_code == 200
        assert response.json() == {"total": 0, "items": []}

# Test case for getting a single collection by ID
def test_get_collection(client: TestClient):
    with patch('services.collection_service.CollectionService.get_collection', return_value=mock_collection):
        response = client.get("/collections/1")
        assert response.status_code == 200
        assert response.json()["name"] == "test_collection"

# Test case for attempting to get a non-existent collection (ID > N)
def test_get_collection_not_found(client: TestClient):
    with patch('services.collection_service.CollectionService.get_collection', side_effect=HTTPException(status_code=404, detail="Collection not found")):
        response = client.get("/collections/999")
        assert response.status_code == 404

# Test case for getting a collection with an ID < 1
def test_get_collection_invalid_id_lt_1(client: TestClient):
    # FastAPI's path validation for integers should handle this.
    # A request to /collections/0 should result in a 404 if not found,
    # as '0' is a valid integer but likely not a valid ID.
    with patch('services.collection_service.CollectionService.get_collection', side_effect=HTTPException(status_code=404, detail="Collection not found")):
        response = client.get("/collections/0")
        assert response.status_code == 404

# Test case for creating a collection with empty embeddings
def test_create_collection_with_empty_embeddings(client: TestClient):
    with patch('services.collection_service.CollectionService.create_collection', return_value=mock_collection):
        response = client.post("/collections", json={
            "name": "test_collection_no_embed",
            "owner": "test_user",
            "description": "A test collection",
            "visibility": "private",
            "embeddings_model": None
        })
        assert response.status_code == 201
        assert response.json()["name"] == "test_collection"