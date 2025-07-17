"""
URL ingestion plugin for web pages.

This plugin handles URLs by fetching their content and processing them into chunks.
Uses Firecrawl Python SDK for web scraping and crawling.
Supports both cloud and local Firecrawl instances.
Uses LangChain's text splitters for text-structured based chunking.
"""

# Python Libraries
import os
import time
from typing import Any, Dict, List, Optional

# Third-Party Libraries
from dotenv import load_dotenv
from firecrawl.firecrawl import FirecrawlApp
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# Local Imports
from .base import IngestPlugin, PluginRegistry

# Load environment variables
load_dotenv()

# Get Firecrawl configuration from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL", "")

@PluginRegistry.register
class URLIngestPlugin(IngestPlugin): 
    """Plugin for ingesting web pages from URLs using Firecrawl."""
    
    name = "url_ingest"
    description = "Ingest web pages from URLs using Firecrawl"
    kind = "base-ingest"
    supported_file_types = {}
    
    def __init__(self):
        """Initialize the plugin with Firecrawl app."""
        super().__init__()
        # Initialize Firecrawl app
        self.firecrawl_app: Optional[FirecrawlApp] = self._init_firecrawl()
    
    def _init_firecrawl(self) -> Optional[FirecrawlApp]:
        """Initialize Firecrawl app. Returns None if configuration is incomplete for cloud or if initialization fails."""
        api_key = FIRECRAWL_API_KEY
        api_url = FIRECRAWL_API_URL
        
        # Ensure API URL has a proper scheme
        if api_url:
            if not (api_url.startswith('http://') or api_url.startswith('https://')):
                api_url = f"https://{api_url}"
                print(f"INFO: [url_ingest] Added https:// scheme to API URL: {api_url}")
        else:
            # Use default API URL if none provided
            api_url = "https://api.firecrawl.dev" # Default to cloud
            print(f"INFO: [url_ingest] FIRECRAWL_API_URL not set, using default API URL: {api_url}")
        
        # Check if API key is required for cloud service
        # The default api_url is cloud, so this check is important
        if 'api.firecrawl.dev' in api_url and not api_key:
            print(f"INFO: [url_ingest] FIRECRAWL_API_KEY not set. API key is required for Firecrawl cloud service ({api_url}). Ingestion will fail if attempted.")
            return None
        
        try:
            # Log configuration
            print(f"INFO: [url_ingest] Attempting to initialize FirecrawlApp with API URL: {api_url}")
            if api_key:
                print(f"INFO: [url_ingest] FIRECRAWL_API_KEY is provided.")
            elif not 'api.firecrawl.dev' in api_url : # No key, and not cloud URL
                 print(f"INFO: [url_ingest] No FIRECRAWL_API_KEY provided; assuming local Firecrawl instance or no key needed for {api_url}.")

            app = FirecrawlApp(api_key=api_key, api_url=api_url)
            print(f"INFO: [url_ingest] FirecrawlApp initialized successfully.")
            return app
        except Exception as e:
            # This catches errors from FirecrawlApp() instantiation, e.g., SDK installed but config issue for FirecrawlApp itself
            print(f"ERROR: [url_ingest] Failed to initialize FirecrawlApp instance: {str(e)}. Check API URL and Firecrawl server status. Ingestion will fail.")
            return None
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters accepted by this plugin.
        
        Returns:
            A dictionary mapping parameter names to their specifications
        """
        return {
            "chunk_size": {
                "type": "integer",
                "description": "Size of each chunk (uses LangChain default if not specified)",
                "default": 2000,
                "required": False
            },
            "chunk_overlap": {
                "type": "integer",
                "description": "Number of units to overlap between chunks (uses LangChain default if not specified)",
                "default": 200,
                "required": False
            },
            "splitter_type": {
                "type": "string",
                "description": "Type of LangChain splitter to use",
                "enum": ["RecursiveCharacterTextSplitter", "CharacterTextSplitter", "TokenTextSplitter"],
                "default": "RecursiveCharacterTextSplitter",
                "required": False
            },
            "urls": {
                "type": "array",
                "description": "List of URLs to ingest",
                "required": True
            }
        }
    
    def ingest(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Ingest URLs and split content into chunks using LangChain's text splitters.
        
        Args:
            file_path: Path to write the processed content
            urls: List of URLs to ingest
            chunk_size: Size of each chunk (default: uses LangChain default)
            chunk_overlap: Number of units to overlap between chunks (default: uses LangChain default)
            splitter_type: Type of LangChain splitter to use (default: RecursiveCharacterTextSplitter)
            
        Returns:
            A list of dictionaries, each containing:
                - text: The chunk text
                - metadata: A dictionary of metadata for the chunk
        """
        if not self.firecrawl_app:
            print("ERROR: [url_ingest] Firecrawl App not initialized. This could be due to a missing API key for the cloud service, the SDK not being installed correctly, or other configuration issues during plugin initialization. Cannot ingest.")
            raise ValueError("Firecrawl App not initialized. Please check plugin initialization logs. If using Firecrawl cloud service, ensure FIRECRAWL_API_KEY is set.")

        # Extract parameters
        chunk_size = kwargs.get("chunk_size", None)
        chunk_overlap = kwargs.get("chunk_overlap", None)
        splitter_type = kwargs.get("splitter_type", "RecursiveCharacterTextSplitter")
        urls = kwargs.get("urls", [])
        
        # Create parameters dict for splitter initialization
        splitter_params = {}
        if chunk_size is not None:
            splitter_params["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            splitter_params["chunk_overlap"] = chunk_overlap
            
        print(f"INFO: [url_ingest] Ingesting {len(urls)} URLs with splitter parameters: {splitter_params}")
        
        if not urls:
            raise ValueError("No URLs provided. Please provide a list of URLs to ingest.")
        
        # Ensure urls is a list
        if isinstance(urls, str):
            urls = [urls]
        
        # Dynamically instantiate the selected LangChain splitter
        try:
            if splitter_type == "RecursiveCharacterTextSplitter":
                text_splitter = RecursiveCharacterTextSplitter(**splitter_params)
            elif splitter_type == "CharacterTextSplitter":
                text_splitter = CharacterTextSplitter(**splitter_params)
            elif splitter_type == "TokenTextSplitter":
                text_splitter = TokenTextSplitter(**splitter_params)
            else:
                raise ValueError(f"Unsupported splitter type: {splitter_type}")
        except ImportError as e:
            raise ImportError(f"Failed to import {splitter_type}: {str(e)}")
        
        all_documents = []
        # To store raw markdown content from all URLs concatenated
        combined_markdown_content = ""
        
        # Always use batch processing, even for a single URL
        print(f"INFO: [url_ingest] Starting batch scrape for {len(urls)} URLs")
        batch_params = {
            'formats': ['markdown']
        }
        
        start_time = time.time()
        batch_result = self.firecrawl_app.batch_scrape_urls(urls, batch_params)
        elapsed_time = time.time() - start_time
        
        print(f"INFO: [url_ingest] Batch scrape completed in {elapsed_time:.2f} seconds")
        print(f"INFO: [url_ingest] Batch response status: {batch_result.get('status')}")
        print(f"INFO: [url_ingest] Batch response data count: {len(batch_result.get('data', []))}")
        
        # Process batch results
        if batch_result and "data" in batch_result and batch_result.get("status") == "completed":
            for i, result in enumerate(batch_result["data"]):
                url = urls[i] if i < len(urls) else "unknown_url"
                try:
                    # Extract markdown content from the result
                    content = None
                    if "markdown" in result:
                        content = result["markdown"]
                        content_length = len(content)
                        print(f"INFO: [url_ingest] URL {url} content extracted: {content_length} chars")
                        # Append content from this URL to the combined string
                        if combined_markdown_content: # Add a separator if not the first content
                            combined_markdown_content += "\n\n---\n\n" # Markdown horizontal rule as separator
                        combined_markdown_content += content
                    
                    if content:
                        print(f"INFO: [url_ingest] Processing content for URL: {url}")
                        # Create base metadata for this URL
                        base_metadata = {
                            "source": url,
                            "filename": url,
                            "extension": "url",
                            "file_size": len(content),
                            "file_url": url,
                            "chunking_strategy": f"langchain_{splitter_type.lower()}"
                        }
                        
                        # Add chunking parameters to metadata if provided
                        if chunk_size is not None:
                            base_metadata["chunk_size"] = chunk_size
                        if chunk_overlap is not None:
                            base_metadata["chunk_overlap"] = chunk_overlap
                        
                        # Split content into chunks using LangChain text splitter
                        chunks = text_splitter.split_text(content)
                        print(f"INFO: [url_ingest] Content split into {len(chunks)} chunks")
                        
                        # Create result documents with metadata
                        for j, chunk_text in enumerate(chunks):
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                "chunk_index": j,
                                "chunk_count": len(chunks)
                            })
                            
                            all_documents.append({
                                "text": chunk_text,
                                "metadata": chunk_metadata
                            })
                    else:
                        print(f"WARNING: [url_ingest] No markdown content found for URL {url} in batch response")
                except Exception as e:
                    print(f"ERROR: [url_ingest] Failed to process batch result for URL {url}: {str(e)}")
                    continue
        else:
            error_msg = f"Batch processing failed with status: {batch_result.get('status', 'unknown')}"
            print(f"ERROR: [url_ingest] {error_msg}")
            raise ValueError(error_msg)
        
        print(f"INFO: [url_ingest] Completed processing for {len(urls)} URLs, generated {len(all_documents)} document chunks")
        
        # Write combined_markdown_content to the provided file path
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(combined_markdown_content)
            print(f"INFO: [url_ingest] Successfully wrote combined markdown content from {len(urls)} URLs to {file_path}")
        except Exception as e:
            print(f"WARNING: [url_ingest] Failed to write combined markdown content to {file_path}: {str(e)}")
            # Optionally, re-raise the exception or handle it as a non-critical error

        return all_documents