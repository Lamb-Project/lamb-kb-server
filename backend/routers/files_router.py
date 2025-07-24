import traceback
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import FileStatus

from dependencies import verify_token

from schemas.files import FileRegistryResponse

from services.collection_service import CollectionService


router = APIRouter(
    prefix="/files",
    tags=["Files"],
    dependencies=[Depends(verify_token)]
)

@router.get(
    "/collection/{collection_id}",
    response_model=List[FileRegistryResponse],
    summary="List files in a collection",
    description="Get a list of all files registered within a specific collection."
)
async def list_files_in_collection(
    collection_id: int,
    db: Session = Depends(get_db),
    status: str = Query(None, description="Filter by status (e.g., completed, processing, failed)")
):
    """
    Lists all files associated with a given collection ID, with an optional filter for status.
    """
    # This keeps the service logic encapsulated
    files = CollectionService.list_files(collection_id, db, status)
    if files is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found."
        )
    return files

@router.put(
    "/{file_id}/status",
    response_model=FileRegistryResponse,
    summary="Update file status",
    description="Update the status of a specific file in the registry."
)
async def update_file_status(
    file_id: int,
    status: str = Query(..., description="New status (e.g., completed, processing, failed)"),
    db: Session = Depends(get_db)
):
    """
    Updates the status of a file registry entry.
    Validates the provided status against the FileStatus enum.
    """
    # Validate that the status is a valid choice
    if status.upper() not in FileStatus.__members__:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{status}'. Valid statuses are: {', '.join(FileStatus.__members__.keys())}"
        )
    
    # The service layer handles the actual update logic
    updated_file = CollectionService.update_file_status(file_id, status.upper(), db)
    if not updated_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found."
        )
    return updated_file

@router.get(
    "/{file_id}/content",
    summary="Get file content",
    description="Get the content of a file from the collection",
    responses={
        200: {"description": "File content retrieved successfully"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "File not found"},
        500: {"description": "Server error"}
    }
)
async def get_file_content(
    file_id: int,
    db: Session = Depends(get_db)
):
    """Get the content of a file."""
    try:
        return CollectionService.get_file_content(file_id, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error retrieving file content: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve file content: {str(e)}"
        )
