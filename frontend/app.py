"""
Streamlit frontend for the Lamb Knowledge Base Server.

This app provides a user interface for:
- Creating and managing collections
- Viewing available ingestion plugins
- Uploading files or ingesting URLs
- Monitoring ingestion status
- Querying collections
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
import os
# from datetime import datetime # Not currently used, can be re-added if needed

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:9090")
API_TOKEN = os.getenv("API_TOKEN", "0p3n-w3bu!")  # Default token for development

# API Client
class KBClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections."""
        response = requests.get(f"{self.base_url}/collections", headers=self.headers)
        response.raise_for_status()
        # Ensure we return the list of items directly
        return response.json().get("items", []) 
    
    def get_collection_details(self, collection_id: int) -> Optional[Dict[str, Any]]:
        """Get details of a specific collection."""
        response = requests.get(f"{self.base_url}/collections/{collection_id}", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return None

    def create_collection(self, name: str, description: Optional[str], visibility: str, owner: str,
                          embeddings_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new collection."""
        data = {
            "name": name,
            "description": description,
            "visibility": visibility,
            "owner": owner
        }
        if embeddings_config:
            # Filter out any None values from embeddings_config before sending
            data["embeddings_model"] = {k: v for k, v in embeddings_config.items() if v is not None and v != ""}
        
        response = requests.post(f"{self.base_url}/collections", headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def list_plugins(self, plugin_type: str = "ingestion") -> List[Dict[str, Any]]:
        """List all available plugins of a given type (ingestion or query)."""
        response = requests.get(f"{self.base_url}/{plugin_type}/plugins", headers=self.headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "items" in data:
             return data.get("items", [])
        elif isinstance(data, list):
            return data
        return [] 
    
    def ingest_file(self, collection_id: int, file, plugin_name: str, plugin_params: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a file into a collection."""
        url = f"{self.base_url}/collections/{collection_id}/ingest-file"
        # Remove Content-Type for multipart/form-data
        headers_for_file_upload = {k: v for k, v in self.headers.items() if k.lower() != 'content-type'}
        
        form_data = {
            'plugin_name': (None, plugin_name),
            'plugin_params': (None, json.dumps(plugin_params))
        }
        
        files = {
            'file': (file.name, file.getvalue(), file.type)
        }
        
        response = requests.post(url, headers=headers_for_file_upload, data=form_data, files=files)
        response.raise_for_status()
        return response.json()
    
    def ingest_urls(self, collection_id: int, urls: List[str], plugin_name: str, plugin_params: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest URLs into a collection."""
        url = f"{self.base_url}/collections/{collection_id}/ingest-url"
        data = {
            "urls": urls,
            "plugin_name": plugin_name,
            "plugin_params": plugin_params
        }
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def ingest_base(self, collection_id: int, plugin_name: str, plugin_params: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest content using a base-ingest plugin.
        
        Args:
            collection_id: ID of the collection
            plugin_name: Name of the base-ingest plugin to use
            plugin_params: Parameters for the plugin
            
        Returns:
            Result of the ingestion operation
        """
        url = f"{self.base_url}/collections/{collection_id}/ingest-base"
        data = {
            "plugin_name": plugin_name,
            "plugin_params": plugin_params
        }
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def list_files(self, collection_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files in a collection."""
        url = f"{self.base_url}/collections/{collection_id}/files"
        if status:
            url += f"?status={status}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        # Assuming this endpoint returns a list directly or a dict with 'items'
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "items" in data:
            return data.get("items", [])
        return []
    
    def query_collection(self, collection_id: int, query_text: str, top_k: int = 5, 
                        threshold: float = 0.0, plugin_name: str = "simple_query",
                        plugin_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query a collection.
        
        Args:
            collection_id: ID of the collection to query
            query_text: The text to query for
            top_k: Number of results to return (default: 5)
            threshold: Minimum similarity threshold (default: 0.0)
            plugin_name: Name of the query plugin to use (default: "simple_query")
            plugin_params: Additional plugin-specific parameters
            
        Returns:
            Query results with timing information
            
        Raises:
            Exception: If the query fails
        """
        if not query_text or query_text.strip() == "":
            raise ValueError("Query text cannot be empty")
            
        plugin_params = plugin_params or {}
        
        # Prepare request data according to QueryRequest schema
        data = {
            "query_text": query_text,
            "top_k": int(top_k),  # Ensure integer
            "threshold": float(threshold),  # Ensure float
            "plugin_params": plugin_params
        }
        
        url = f"{self.base_url}/collections/{collection_id}/query?plugin_name={plugin_name}" # Define URL
        
        try:
            # Use requests.post directly
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)
            return response.json()  # Return JSON response
        except requests.exceptions.HTTPError as http_err:
            # Try to get more detail from response if possible
            error_detail = str(http_err)
            if http_err.response is not None:
                try:
                    error_detail = http_err.response.json().get('detail', str(http_err))
                except json.JSONDecodeError: # If response is not JSON
                    error_detail = http_err.response.text # Fallback to raw text
            raise Exception(f"Query failed: {error_detail}")
        except Exception as e:
            # Catch other potential errors (network issues, etc.)
            raise Exception(f"Query failed: {str(e)}")

# Initialize the client
client = KBClient(API_BASE_URL, API_TOKEN)

# Initialize session state
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'collections_list'
if 'selected_collection_id' not in st.session_state:
    st.session_state.selected_collection_id = None
if 'selected_collection_name' not in st.session_state:
    st.session_state.selected_collection_name = None
if 'collection_doc_counts' not in st.session_state:
    st.session_state.collection_doc_counts = {}
if 'query_results' not in st.session_state:
    st.session_state.query_results = None # To store query results

st.set_page_config(page_title="Lamb Knowledge Base", layout="wide")
st.title("🐑 Lamb Knowledge Base")

# --- Helper function to switch views ---
def change_view(view_name, collection_id=None, collection_name=None):
    st.session_state.current_view = view_name
    st.session_state.selected_collection_id = collection_id
    st.session_state.selected_collection_name = collection_name
    st.session_state.query_results = None # Reset query results when changing views or collections

# --- Helper function to get total document count for a collection ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_total_document_count(_client, collection_id: int) -> int:
    """Fetches files for a collection and sums their document counts."""
    # st.write(f"Fetching doc count for {collection_id}") # For debugging
    try:
        files = _client.list_files(collection_id)
        total_docs = sum(file.get('document_count', 0) for file in files)
        return total_docs
    except Exception as e:
        # st.error(f"Error getting doc count for {collection_id}: {e}") # For debugging
        return 0 # Return 0 on error to avoid breaking UI

# --- Views ---

def display_collections_list():
    st.header("Collections")
    
    with st.expander("Create New Collection"):
        with st.form("create_collection_form"):
            st.subheader("Basic Information")
            name = st.text_input("Collection Name *")
            owner = st.text_input("Owner *") # Added Owner
            description = st.text_area("Description")
            visibility = st.selectbox("Visibility", ["private", "public"], index=0)
            
            st.subheader("Embedding Configuration")
            st.caption("Specify 'default' to use server-configured environment variables for model, vendor, endpoint, or API key.")
            
            embed_model_name = st.text_input("Model Name", value="default", help="e.g., sentence-transformers/all-MiniLM-L6-v2, text-embedding-3-small, or 'default'")
            embed_vendor = st.selectbox("Vendor", ["default", "local", "openai", "ollama", "clip"], index=0, help="e.g., 'local', 'openai', 'clip', 'ollama', or 'default'")
            embed_api_endpoint = st.text_input("API Endpoint (Optional)", value="default", help="e.g., http://localhost:11434 for Ollama, or 'default'")
            embed_apikey = st.text_input("API Key (Optional)", type="password", value="default", help="Required for some vendors like OpenAI, or 'default'")

            submitted = st.form_submit_button("Create Collection")
            if submitted:
                if not name or not owner:
                    st.error("Collection Name and Owner are required fields.")
                else:
                    embeddings_config = {
                        "model": embed_model_name,
                        "vendor": embed_vendor,
                        "api_endpoint": embed_api_endpoint if embed_api_endpoint and embed_api_endpoint.lower() != 'default' else None,
                        "apikey": embed_apikey if embed_apikey and embed_apikey.lower() != 'default' else None
                    }
                    # Filter out None values, but keep 'default' strings if they are explicitly set to default
                    # The backend handles 'default' strings. Empty strings for optional fields should be treated as None.
                    if embeddings_config["api_endpoint"] == "": embeddings_config["api_endpoint"] = None
                    if embeddings_config["apikey"] == "": embeddings_config["apikey"] = None
                    
                    # If all embedding fields are effectively 'default' or None, send None for embeddings_model to backend
                    # so backend uses its complete default logic.
                    # Otherwise, send the specific config.
                    final_embeddings_config = None
                    if not (embed_model_name.lower() == 'default' and 
                            embed_vendor.lower() == 'default' and 
                            (embeddings_config["api_endpoint"] is None or embed_api_endpoint.lower() == 'default') and
                            (embeddings_config["apikey"] is None or embed_apikey.lower() == 'default')):
                        final_embeddings_config = embeddings_config

                    try:
                        result = client.create_collection(name, description, visibility, owner, embeddings_config=final_embeddings_config)
                        st.success(f"Collection '{result['name']}' created successfully!")
                        st.cache_data.clear() # Clear cache to refresh doc counts on next load
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating collection: {str(e)}")
                        st.exception(e)
    
    st.subheader("Your Collections")
    # Add a button to refresh document counts manually if needed
    if st.button("Refresh Document Counts"):
        st.cache_data.clear()
        st.rerun()
        
    try:
        collections = client.list_collections()
        if collections:
            cols = st.columns(3)
            for i, collection in enumerate(collections):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.subheader(collection['name'])
                        st.caption(f"ID: {collection['id']} | Owner: {collection.get('owner', 'N/A')} | Vis: {collection['visibility']}")
                        st.write(collection.get('description', 'No description provided.'))
                        st.text(f"Created: {collection.get('creation_date', 'N/A')}")
                        
                        embed_info = collection.get('embeddings_model', {})
                        st.markdown("**Embeddings:**")
                        model_details = []
                        if embed_info.get('model'): model_details.append(f"Model: `{embed_info['model']}`")
                        if embed_info.get('vendor'): model_details.append(f"Vendor: `{embed_info['vendor']}`")
                        if embed_info.get('api_endpoint'): model_details.append(f"Endpoint: `{embed_info['api_endpoint']}`")
                        if model_details:
                            st.markdown("\n".join(f"- {detail}" for detail in model_details))
                        else:
                            st.markdown("- Not specified")
                            
                        # Fetch and display document count
                        # This will be slow for many collections due to N+1 calls.
                        # A spinner can be added here if desired.
                        with st.spinner(f"Counting docs for {collection['name']}..."):
                             doc_count = get_total_document_count(client, collection['id'])
                        st.metric(label="Documents", value=doc_count)
                        
                        if st.button("View Details", key=f"view_{collection['id']}", use_container_width=True):
                            change_view('collection_detail', collection_id=collection['id'], collection_name=collection['name'])
                            st.rerun()
        else:
            st.info("No collections found. Create your first collection above!")
    except Exception as e:
        st.error(f"Error loading collections: {str(e)}")
        st.exception(e) # show full traceback for debugging

def display_collection_detail():
    collection_id = st.session_state.selected_collection_id
    collection_name = st.session_state.selected_collection_name

    if collection_id is None:
        st.error("No collection selected.")
        if st.button("Go back to Collections List"):
            change_view('collections_list')
            st.rerun()
        return

    st.header(f"Collection: {collection_name}")
    if st.button("⬅️ Back to Collections List"):
        change_view('collections_list')
        st.rerun()

    # Tabs for different actions within a collection
    tab_titles = ["View Files", "Ingest Content", "Query Collection"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    with tab1:
        st.subheader("Files in this Collection")

        if st.button("🔄 Refresh Files", key=f"refresh_files_{collection_id}"):
            st.rerun() # Rerun the app to refresh the file list

        try:
            status_filter = st.selectbox(
                "Filter by Status",
                ["all", "completed", "processing", "failed"],
                index=0,
                format_func=lambda x: x.capitalize(),
                key=f"status_filter_{collection_id}"
            )
            files = client.list_files(collection_id, status=None if status_filter == "all" else status_filter)
            if files:
                for file_item in files:
                    with st.expander(f"{file_item.get('original_filename', 'N/A')} ({file_item.get('status', 'N/A')})"):
                        st.json(file_item) # Display full file info for now
            else:
                st.info("No files found in this collection for the selected status.")
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            st.exception(e)

    with tab2:
        st.subheader("Ingest New Content")
        try:
            plugins = client.list_plugins(plugin_type="ingestion")
            if not plugins:
                st.warning("No ingestion plugins available.")
                return

            plugin_options = {p['name']: p for p in plugins}
            selected_plugin_name = st.selectbox(
                "Select Ingestion Plugin",
                options=list(plugin_options.keys()),
                format_func=lambda x: f"{x} - {plugin_options[x].get('description', 'No description')}",
                key=f"ingestion_plugin_select_{collection_id}"
            )
            
            selected_plugin = plugin_options[selected_plugin_name]
            
            st.markdown("**Plugin Parameters**")
            # Collect user inputs for parameters
            user_plugin_inputs = {}
            with st.form(key=f"ingest_form_{collection_id}_{selected_plugin_name}"):
                for param_name, param_info in selected_plugin.get('parameters', {}).items():
                    field_key = f"param_ingest_{param_name}_{collection_id}_{selected_plugin_name}"
                    default_value = param_info.get('default')
                    user_input = None
                    if param_info['type'] == 'integer':
                        user_input = st.number_input(
                            param_name,
                            value=default_value if default_value is not None else 0,
                            help=param_info.get('description'),
                            key=field_key
                        )
                    elif param_info['type'] == 'string':
                        if 'enum' in param_info:
                            idx = 0
                            if default_value and default_value in param_info['enum']:
                                idx = param_info['enum'].index(default_value)
                            user_input = st.selectbox(
                                param_name,
                                options=param_info['enum'],
                                index=idx,
                                help=param_info.get('description'),
                                key=field_key
                            )
                        else:
                            user_input = st.text_input(
                                param_name,
                                value=default_value if default_value is not None else '',
                                help=param_info.get('description'),
                                key=field_key
                            )
                    elif param_info['type'] == 'array' and param_name == 'urls':
                        user_input = st.text_area(
                            param_name,
                            value="\n".join(default_value if default_value is not None else []),
                            help=param_info.get('description', "Enter URLs, one per line"),
                            height=100,
                            key=field_key
                        )
                    elif param_info['type'] == 'boolean':
                        user_input = st.checkbox(
                            param_name,
                            value=default_value if default_value is not None else False,
                            help=param_info.get('description'),
                            key=field_key
                        )
                    elif param_info['type'] == 'enum':
                        options = param_info.get('options', [])
                        idx = 0
                        if default_value in options:
                            idx = options.index(default_value)
                        user_input = st.selectbox(
                            param_name,
                            options=options,
                            index=idx,
                            help=param_info.get('description'),
                            key=field_key
                            )
                    user_plugin_inputs[param_name] = user_input

                ingest_button_label = "Ingest"
                uploaded_file = None
                if selected_plugin.get('kind') == 'file-ingest':
                    uploaded_file = st.file_uploader("Choose a file", type=None, key=f"uploader_{collection_id}_{selected_plugin_name}")
                    ingest_button_label = "Ingest File"
                elif selected_plugin.get('kind') == 'base-ingest':
                    ingest_button_label = "Process Content"

                form_submitted = st.form_submit_button(ingest_button_label)
                if form_submitted:
                    # Prepare plugin_params, excluding those that match their default value
                    params_for_ingestion = {}
                    for param_name, user_value in user_plugin_inputs.items():
                        param_info = selected_plugin.get('parameters', {}).get(param_name)
                        if param_info:
                            default_from_plugin = param_info.get('default')
                            param_type = param_info.get('type')

                            send_parameter = False
                            if param_type == 'array' and param_name == 'urls':
                                actual_user_value = [url.strip() for url in user_value.split('\n') if url.strip()]
                                # UI default for URLs (plugin_default=None) is an empty list
                                default_list_for_urls = default_from_plugin if default_from_plugin is not None else []
                                if actual_user_value != default_list_for_urls:
                                    send_parameter = True
                                    # Value to send is actual_user_value for URLs
                                    params_for_ingestion[param_name] = actual_user_value 
                            else: # For non-URL parameters (string, integer, enum/selectbox, etc.)
                                if default_from_plugin is not None:
                                    # If there's an explicit default from plugin (e.g., "abc", 123)
                                    if user_value != default_from_plugin:
                                        send_parameter = True
                                else:
                                    # If plugin's default is None (or not specified)
                                    if param_type == 'string':
                                        # UI default for string (plugin_default=None) is ''
                                        if user_value != '': # Send only if user typed something
                                            send_parameter = True
                                    elif param_type == 'integer':
                                        # UI default for integer (plugin_default=None) is 0
                                        if user_value != 0: # Send only if user changed from 0
                                            send_parameter = True
                                    # For other types like enums from selectbox:
                                    # If default_from_plugin is None, and user_value is also None (if widget allows),
                                    # then user_value != default_from_plugin would be false, so not sent by previous logic.
                                    # If user_value is not None (e.g., a selected enum value), then it should be sent.
                                    # This covers selectboxes where the UI default might be the first enum if plugin default is None.
                                    elif user_value is not None: # Send if user_value is anything other than None
                                        send_parameter = True
                            
                            if send_parameter and not (param_type == 'array' and param_name == 'urls'):
                                # Add to params if send_parameter is true, unless it's URLs (already added)
                                params_for_ingestion[param_name] = user_value
                        else: # Should not happen if user_plugin_inputs is built correctly from plugin params
                            params_for_ingestion[param_name] = user_value # Fallback: send it anyway
                    try:
                        if selected_plugin.get('kind') == 'file-ingest':
                            if uploaded_file:
                                result = client.ingest_file(collection_id, uploaded_file, selected_plugin_name, params_for_ingestion)
                                st.success(f"File '{uploaded_file.name}' ingestion started: {result.get('message', 'Check status in View Files tab.')}")
                            else:
                                st.error("Please upload a file.")
                        elif selected_plugin.get('kind') == 'base-ingest':
                            # For base-ingest plugins, we use the new ingest_base endpoint
                            result = client.ingest_base(collection_id, selected_plugin_name, params_for_ingestion)
                            st.success(f"Content processing started: {result.get('message', 'Check status in View Files tab.')}")
                        else:
                            st.warning(f"Ingestion kind '{selected_plugin.get('kind')}' not fully supported for direct submission yet.")
                        
                    except Exception as e:
                        st.error(f"Error during ingestion: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"Error loading ingestion plugins or parameters: {str(e)}")
            st.exception(e)

    with tab3:
        st.subheader("Query this Collection")
        query_text = st.text_area("Enter your query:", key=f"query_text_{collection_id}")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.number_input("Top K results:", min_value=1, max_value=100, value=5, key=f"query_top_k_{collection_id}")
        with col2:
            threshold = st.slider("Similarity Threshold:", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key=f"query_threshold_{collection_id}")
        
        if st.button("Search", key=f"query_button_{collection_id}"):
            if query_text:
                try:
                    with st.spinner("Searching..."):
                        results_data = client.query_collection(collection_id, query_text, top_k, threshold)
                        st.session_state.query_results = results_data.get("results", [])
                except Exception as e:
                    st.error(f"Error during query: {str(e)}")
                    st.exception(e)
                    st.session_state.query_results = [] # Clear on error
            else:
                st.warning("Please enter a query.")
                st.session_state.query_results = [] # Clear if no text

        if st.session_state.query_results is not None:
            if not st.session_state.query_results:
                st.info("No results found for your query, or query was empty.")
            else:
                st.markdown(f"**Found {len(st.session_state.query_results)} results:**")
                for i, result in enumerate(st.session_state.query_results):
                    with st.container(border=True):
                        st.markdown(f"**Result {i+1} (Similarity: {result.get('similarity', 0.0):.4f})**")
                        metadata = result.get('metadata', {})
                        source_info = []
                        if metadata.get('source'): source_info.append(f"Source: `{metadata['source']}`")
                        if metadata.get('filename'): source_info.append(f"File: `{metadata['filename']}`")
                        if metadata.get('chunk_index') is not None: source_info.append(f"Chunk: {metadata['chunk_index']}")
                        if source_info:
                            st.markdown(", ".join(source_info))
                        
                        st.text_area(f"Content_{i}", value=result.get('data', ''), height=150, disabled=True, key=f"res_data_{collection_id}_{i}")
                        with st.expander("View Full Metadata"):
                            st.json(metadata)

# --- Main app router ---
if st.session_state.current_view == 'collections_list':
    display_collections_list()
elif st.session_state.current_view == 'collection_detail':
    display_collection_detail()
else:
    st.error("Invalid view state. Resetting to collections list.")
    change_view('collections_list')
    st.rerun() 