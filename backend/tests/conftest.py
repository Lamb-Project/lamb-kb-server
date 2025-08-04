# backend/tests/conftest.py

import pytest
import os
from starlette.testclient import TestClient

# Import the main app and the dependency you want to override
from main import app
from dependencies import verify_token

# This is a dummy function that will replace the real 'verify_token'
async def override_verify_token():
    return {}

# Apply the override to the FastAPI app instance for all tests
app.dependency_overrides[verify_token] = override_verify_token


@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """
    Set up the necessary environment variables for the entire test session.
    """
    os.environ["EMBEDDINGS_VENDOR"] = "ollama"
    os.environ["EMBEDDINGS_MODEL"] = "nomic-embed-text"
    yield
    # Clean up after all tests have run
    del os.environ["EMBEDDINGS_VENDOR"]
    del os.environ["EMBEDDINGS_MODEL"]


@pytest.fixture(scope="module")
def client():
    """
    Provide a TestClient instance for the tests.
    """
    with TestClient(app) as c:
        yield c