# Lamb Knowledge Base Server (lamb-kb-server)

A dedicated knowledge base server designed to provide robust vector database functionality for the LAMB project and to serve as a Model Context Protocol (MCP) server. It uses ChromaDB for vector database storage and FastAPI to create an API that allows the LAMB project to access knowledge databases.

**Authors:** [Marc Alier (@granludo)](https://github.com/granludo) and [Juanan Pereira (@juananpe)](https://github.com/juananpe)

> **Note:** The Model Context Protocol (MCP) functionality is currently a work in progress.

## Setup and Installation

### Option 1: Local Installation
#### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- recommended use of Conda or virtual environment 

#### Installation

1. Clone the repository:
```bash
git clone https://github.com/Lamb-Project/lamb-kb-server.git
cd lamb-kb-server
```

2. Set-up and install the required dependencies:
```bash
cd backend
mkdir static
pip install -r requirements.txt
```

   > **Project Structure:** The repository consists of a `backend` directory that contains the server code. All commands should be run from the `backend` directory.

3. Environment variables:
   - Copy `.env.example` to `.env`
   - Modify the API key as needed (default: "0p3n-w3bu!")
   - Configure embedding model settings (see Embeddings Configuration section)

#### Running the Server

```bash
cd backend
python start.py
```

The server will run on http://localhost:9090 by default. Edit start.py to change the port. 

#### Web Explorer and Diagnostics

The Lamb KB Server includes a web-based UI to explore collections and diagnose potential issues with the ChromaDB database:

```bash
cd backend
python lamb_kb_webapp.py
```

This starts a web server on http://localhost:5000 by default that provides:

1. **Collection Explorer**: Browse, view, and query your knowledge base collections
2. **Advanced Diagnostics**: Troubleshoot issues between the SQLite registry and ChromaDB storage
   - Collection mappings verification
   - UUID consistency checks
   - Metadata analysis
   - Segments information
   - Detection of orphaned data

The diagnostics tool is especially useful when:
- Collections don't appear correctly in the API
- You encounter "collection not found" errors
- Documents aren't being properly stored or retrieved
- You need to verify database integrity

![Diagnostics Screenshot]

The diagnostics page helps identify and fix common issues like:
- Case sensitivity mismatches in collection names
- Missing UUID mappings
- Orphaned data directories
- Inconsistencies between the SQLite registry and ChromaDB 

### Option 2: Docker Containers
1. Clone the repository:
```bash
git clone git@github.com:Lamb-Project/lamb-kb-server.git
```
2. Rename the ``.env.example``  to ``.env``.
3. Execute ``docker-compose -f docker-compose.dev.yaml up -d`` for development and ``docker-compose -f docker-compose.prod.yaml up -d`` for production.
4. You have a ready development/production environment running.
5. Running unit and integration tests: ``docker compose -f docker-compose.dev.yaml --env-file backend/.env run --rm backend-tests /opt/deps/bin/pytest tests/``

**NOTE**: development uses only CPU. If you want to use the GPU you need to modify the docker-compose.prod.yaml file accordingly to the specific GPU you have.

## API Authentication

All API calls require a Bearer token for authentication. The token must match the `LAMB_API_KEY` environment variable.

Example request:
```bash
curl -H 'Authorization: Bearer 0p3n-w3bu!' http://localhost:9090/
```

## Features

### Core Functionality

- **Collections Management**: Create, view, update and manage document collections
- **Document Ingestion**: Process and store documents with vectorized content
- **File Registry**: Maintain a registry of all uploaded and processed files with their metadata and processing status
- **Similarity Search**: Query collections to find semantically similar content
- **Static File Serving**: Serve original documents via URL references

### Plugin System

The server implements a flexible plugin architecture for both ingestion and querying:

#### Ingestion Plugins

Plugins for processing different document types with configurable chunking strategies:

- **simple_ingest**: Processes text files with options for character, word, or line-based chunking
- Support for custom chunking parameters:
  - `chunk_size`: Size of each chunk
  - `chunk_unit`: Unit for chunking (`char`, `word`, or `line`)
  - `chunk_overlap`: Overlap between chunks

#### Query Plugins

Plugins for different query strategies:

- **simple_query**: Performs similarity searches with configurable parameters:
  - `top_k`: Number of results to return
  - `threshold`: Minimum similarity threshold

### Embeddings Configuration

The system supports multiple embedding providers:

1. **Local Embeddings** (default)
   - Uses sentence-transformers models locally
   - Example configuration in `.env`:
     ```
     EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
     EMBEDDINGS_VENDOR=local
     EMBEDDINGS_APIKEY=
     ```

2. **OpenAI Embeddings**
   - Uses OpenAI's embedding API
   - Requires an API key
   - Example configuration in `.env`:
     ```
     EMBEDDINGS_MODEL=text-embedding-3-small
     EMBEDDINGS_VENDOR=openai
     EMBEDDINGS_APIKEY=your-openai-key-here
     ```

When creating collections, you can specify the embedding configuration or use "default" to inherit from environment variables:

```json
"embeddings_model": {
  "model": "default",
  "vendor": "default",
  "apikey": "default"
}
```

## API Examples

Full api documentation is available at http://localhost:9090/docs (swagger) and backend/lamb-kb-server-api.md (markdown, wich can be generated with the backend/export_docs.py script ).

### Creating a Collection

```bash
curl -X POST 'http://localhost:9090/collections' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my-knowledge-base",
    "description": "My first knowledge base",
    "owner": "user1",
    "visibility": "private",
    "embeddings_model": {
      "model": "default",
      "vendor": "default",
      "apikey": "default"
    }
  }'
```

### Ingesting a File

```bash
curl -X POST 'http://localhost:9090/collections/1/ingest_file' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -F 'file=@/path/to/document.txt' \
  -F 'plugin_name=simple_ingest' \
  -F 'plugin_params={"chunk_size":1000,"chunk_unit":"char","chunk_overlap":200}'
```

### Querying a Collection

```bash
curl -X POST 'http://localhost:9090/collections/1/query' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -H 'Content-Type: application/json' \
  -d '{
    "query_text": "What is machine learning?",
    "top_k": 5,
    "threshold": 0.5,
    "plugin_params": {}
  }'
```

## Testing

The repository includes test scripts to verify functionality:

- **test.py**: A comprehensive test script that tests every API endpoint including creating collections, ingesting documents with different chunking strategies, performing queries, and file registry management
- **params.json**: Configuration for the test script with example data and queries

The test script provides a complete workflow example using the `LambKBClient` class, which can be used as a reference for integrating with the API in your own projects.

To run the tests:

```bash
cd backend
python test.py
```

## File Registry

The system maintains a registry of all uploaded and processed files with their metadata and processing status:

- **List Files**: View all files in a collection or filter by status
- **File Status Management**: Track and update file status (processing, completed, failed, deleted)
- **File Metadata**: Access metadata such as original filename, size, processing statistics, and chunking strategy

Example API calls for file registry management:

```bash
# List all files in a collection
curl -X GET 'http://localhost:9090/collections/1/files' \
  -H 'Authorization: Bearer 0p3n-w3bu!'

# Update file status
curl -X PATCH 'http://localhost:9090/files/1' \
  -H 'Authorization: Bearer 0p3n-w3bu!' \
  -H 'Content-Type: application/json' \
  -d '{"status": "completed"}'
```

# KNOWN ISSUES
[Bug](https://github.com/chroma-core/chroma/issues/2731): Chroma 0.5.5 not working on intel CPU macbook #2731
chroma-core/chroma#2731

# Web Application (just for development and debugging)

The repository includes a web application that can be used to explore collections and diagnose potential issues with the ChromaDB database and consistency with the sqlite lamb-kb-server.db database.

To run the web application:

```bash
  cd backend
  python lamb_kb_webapp.py
  ```

  
go to http://localhost:9091 to access the web application 
Warning, this app runs on port 9091, different from the server port 9090 
Warning, this app is not production ready and is only for development and debugging purposes 




