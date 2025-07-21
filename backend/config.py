"""
Default configuration settings for the Lamb Knowledge Base Server.
"""

import os

API_KEY = os.getenv("LAMB_API_KEY", "0p3n-w3bu!")

DEFAULT_EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
DEFAULT_EMBEDDINGS_VENDOR = os.getenv("EMBEDDINGS_VENDOR", "ollama")
DEFAULT_EMBEDDINGS_APIKEY = os.getenv("EMBEDDINGS_APIKEY", "")
DEFAULT_EMBEDDINGS_ENDPOINT = os.getenv("EMBEDDINGS_ENDPOINT", "http://localhost:11434/api/embeddings")

# For development, allow all origins. For production, you should restrict
# this to your frontend's domain.
# Example: CORS_ORIGINS = ["http://localhost:3000", "https://your-frontend.com"]
CORS_ORIGINS = ["*"]