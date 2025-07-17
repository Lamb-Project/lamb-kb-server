"""
Perplexity Sonar ingestion plugin.

This plugin interfaces with the Perplexity Sonar API to perform deep research
based on user queries and processes the results into chunks.
"""

# Python Libraries
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-Party Libraries
from dotenv import load_dotenv
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
import requests

# Local Imports
from .base import IngestPlugin, PluginRegistry

# Load environment variables
load_dotenv()

# Get Perplexity configuration from environment variables
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

@PluginRegistry.register
class PerplexitySonarPlugin(IngestPlugin):
    """Plugin for ingesting research data from Perplexity Sonar API."""

    name = "perplexity_sonar_ingest"
    description = "Ingest research data from Perplexity Sonar API based on a query"
    kind = "base-ingest"
    supported_file_types = {}  # Takes a query, not a file

    def __init__(self):
        """Initialize the plugin with the Perplexity client."""
        super().__init__()
        self.perplexity_config = self._init_perplexity_config()

    def _init_perplexity_config(self) -> Optional[Dict[str, str]]:
        """Initialize Perplexity API configuration."""
        if not PERPLEXITY_API_KEY:
            print("INFO: [perplexity_sonar_plugin] PERPLEXITY_API_KEY not set in environment variables. API key will be required for ingestion.")
            return None
        
        config = {
            "api_key": PERPLEXITY_API_KEY,
            "chat_completions_url": "https://api.perplexity.ai/chat/completions"
        }
        print(f"INFO: [perplexity_sonar_plugin] Perplexity API config initialized for URL: {config['chat_completions_url']}")
        return config
        # No actual client object to return, just config. Error cases are handled by raising ValueError.

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameters accepted by this plugin.

        Returns:
            A dictionary mapping parameter names to their specifications.
        """
        return {
            "query": {
                "type": "string",
                "description": "The research query or question for Perplexity Sonar API.",
                "required": True
            },
            "model_name": {
                "type": "string",
                "description": "Perplexity model to use for research. 'sonar-deep-research' is for exhaustive reports, 'sonar-pro' for advanced search, and 'sonar' for lighter search.",
                "enum": ["sonar-deep-research", "sonar-pro", "sonar"],
                "default": "sonar-deep-research", 
                "required": False
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature, between 0 and 2. Higher values make output more random. (Optional)",
                "required": False
            },
            # Parameters for chunking, similar to url_ingest
            "chunk_size": {
                "type": "integer",
                "description": "Size of each chunk for the research result (uses LangChain default if not specified).",
                "default": 2000,
                "required": False
            },
            "chunk_overlap": {
                "type": "integer",
                "description": "Number of units to overlap between chunks (uses LangChain default if not specified).",
                "default": 200,
                "required": False
            },
            "splitter_type": {
                "type": "string",
                "description": "Type of LangChain splitter to use for the research result.",
                "enum": ["RecursiveCharacterTextSplitter", "CharacterTextSplitter", "TokenTextSplitter"],
                "default": "RecursiveCharacterTextSplitter",
                "required": False
            }
        }

    def ingest(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """Ingest data from Perplexity Sonar API based on a query and split content into chunks.

        Args:
            file_path: Path to write the full processed content from Perplexity.
            **kwargs: Plugin parameters including 'query', 'model_name', chunking params, etc.

        Returns:
            A list of dictionaries, each containing:
                - text: The chunk text.
                - metadata: A dictionary of metadata for the chunk.
        """
        if not self.perplexity_config:
            print("ERROR: [perplexity_sonar_plugin] Perplexity API key not configured. Cannot ingest.")
            raise ValueError("Perplexity API key not configured. Please set PERPLEXITY_API_KEY environment variable.")

        # Extract parameters
        query = kwargs.get("query")
        if not query:
            raise ValueError("No query provided. Please provide a 'query' for Perplexity Sonar.")

        model_name = kwargs.get("model_name", "sonar-deep-research") # Default from get_parameters
        temperature = kwargs.get("temperature")
        
        chunk_size = kwargs.get("chunk_size") # Default handled by splitter if None
        chunk_overlap = kwargs.get("chunk_overlap") # Default handled by splitter if None
        splitter_type = kwargs.get("splitter_type", "RecursiveCharacterTextSplitter")

        print(f"INFO: [perplexity_sonar_plugin] Processing query: '{query}' using model: {model_name}")

        # Prepare API call parameters
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI research assistant. Provide comprehensive and well-structured information "
                    "based on the user's query. Focus on factual accuracy and depth."
                ),
            },
            {
                "role": "user", 
                "content": query
            },
        ]
        
        api_params = {
            "model": model_name,
            "messages": messages,
        }
        if temperature is not None:
            api_params["temperature"] = temperature
            
        headers = {
            "Authorization": f"Bearer {self.perplexity_config['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
            
        # Make the API call to Perplexity
        try:
            print(f"INFO: [perplexity_sonar_plugin] Sending request to Perplexity API: {self.perplexity_config['chat_completions_url']} with payload: {json.dumps(api_params)}")
            
            http_response = requests.post(
                self.perplexity_config['chat_completions_url'],
                headers=headers,
                json=api_params # requests library handles json.dumps internally for the json parameter
            )
            http_response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            
            response_data = http_response.json()
            
            # Extract the content from the response
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message", {})
                research_content = message.get("content", "")
                if not research_content:
                     print(f"WARNING: [perplexity_sonar_plugin] 'content' field missing or empty in API response choice message: {message}")
                full_api_response_json = json.dumps(response_data, indent=2)
            else:
                print(f"ERROR: [perplexity_sonar_plugin] No 'choices' found or 'choices' array empty in Perplexity API response: {response_data}")
                research_content = ""
                full_api_response_json = json.dumps(response_data, indent=2) if response_data else "{}"

        except requests.exceptions.HTTPError as http_err:
            error_content = http_err.response.text if http_err.response else "No response content"
            print(f"ERROR: [perplexity_sonar_plugin] HTTP error occurred during API call to Perplexity: {http_err} - {error_content}")
            research_content = f"Error during Perplexity API call (HTTP {http_err.response.status_code if http_err.response else 'Unknown'}): {error_content}"
            full_api_response_json = json.dumps({"error": str(http_err), "response_text": error_content})
        except requests.exceptions.RequestException as req_err:
            print(f"ERROR: [perplexity_sonar_plugin] Request exception occurred during API call to Perplexity: {req_err}")
            research_content = f"Error during Perplexity API call (Request Exception): {str(req_err)}"
            full_api_response_json = json.dumps({"error": str(req_err)})
        except Exception as e: # Catch any other unexpected errors, e.g., JSON parsing if content-type is wrong
            print(f"ERROR: [perplexity_sonar_plugin] An unexpected error occurred during API call or processing: {str(e)}")
            research_content = f"Unexpected error during Perplexity API interaction: {str(e)}"
            full_api_response_json = json.dumps({"error": str(e)})


        # Save the full research content to the specified file_path
        try:
            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(output_path, 'w', encoding='utf-8') as f:
                # You might want to save the raw JSON response or just the extracted content
                # For now, saving the primary content. Could also save full_api_response_json.
                f.write(research_content)
            print(f"INFO: [perplexity_sonar_plugin] Full research content saved to: {file_path}")
        except Exception as e:
            print(f"ERROR: [perplexity_sonar_plugin] Failed to write research content to {file_path}: {str(e)}")

        # --- Chunking Logic (similar to url_ingest) ---
        all_documents = []
        if research_content:
            # Initialize LangChain splitter
            splitter_params = {}
            if chunk_size is not None:
                splitter_params["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                splitter_params["chunk_overlap"] = chunk_overlap
            
            try:
                if splitter_type == "RecursiveCharacterTextSplitter":
                    text_splitter = RecursiveCharacterTextSplitter(**splitter_params)
                elif splitter_type == "CharacterTextSplitter":
                    text_splitter = CharacterTextSplitter(**splitter_params)
                elif splitter_type == "TokenTextSplitter":
                    # TokenTextSplitter might require a model name or encoding
                    # For now, assuming basic usage; might need adjustment
                    splitter_params.pop("chunk_overlap", None) # TokenTextSplitter might not use overlap in the same way or at all
                    text_splitter = TokenTextSplitter(**splitter_params)
                else:
                    raise ValueError(f"Unsupported splitter type: {splitter_type}")
                
                chunks = text_splitter.split_text(research_content)
                print(f"INFO: [perplexity_sonar_plugin] Research content split into {len(chunks)} chunks using {splitter_type}")

                base_metadata = {
                    "source": "perplexity_sonar",
                    "query": query,
                    "model_name": model_name,
                    "file_path_original_full_output": str(file_path), # Link to the saved full output
                    "chunking_strategy": f"langchain_{splitter_type.lower().replace('textsplitter', '')}"
                }
                if chunk_size is not None:
                    base_metadata["chunk_size"] = chunk_size
                if chunk_overlap is not None and splitter_type != "TokenTextSplitter": # only add if relevant
                    base_metadata["chunk_overlap"] = chunk_overlap

                for i, chunk_text in enumerate(chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "chunk_count": len(chunks)
                    })
                    all_documents.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })

            except Exception as e:
                print(f"ERROR: [perplexity_sonar_plugin] Failed to split research content: {str(e)}")


        print(f"INFO: [perplexity_sonar_plugin] Completed processing for query, generated {len(all_documents)} document chunks.")
        return all_documents