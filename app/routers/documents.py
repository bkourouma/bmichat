"""
BMI Chat Application - Documents Router

This module provides document management endpoints for uploading,
processing, and managing documents in the knowledge base.
"""

from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from loguru import logger

from app.config import settings
from app.core.database import get_db
from app.core.exceptions import DocumentProcessingError, ValidationError
from app.services.document_manager import DocumentManager


router = APIRouter()

# Initialize document manager
document_manager = DocumentManager()


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type/extension")
    file_size: int = Field(..., description="File size in bytes")
    upload_date: str = Field(..., description="Upload timestamp")
    status: str = Field(..., description="Processing status")
    chunk_count: Optional[int] = Field(None, description="Number of chunks created")


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")


@router.post("/documents/upload", response_model=DocumentInfo, summary="Upload document")
async def upload_document(
    file: UploadFile = File(...),
    keywords: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> DocumentInfo:
    """
    Upload and process a document for the knowledge base.
    
    Args:
        file: Uploaded file
        keywords: Optional keywords for categorization (e.g., "product", "service", "policy")
        db: Database session

    Returns:
        DocumentInfo with upload details
        
    Raises:
        HTTPException: If upload or processing fails
    """
    try:
        logger.info(f"📄 Uploading document: {file.filename}")

        # Read file content
        content = await file.read()

        # Upload and process document
        document = await document_manager.upload_document(
            file_content=content,
            filename=file.filename,
            db_session=db,
            keywords=keywords
        )

        # Convert to response format
        document_info = DocumentInfo(
            id=document.id,
            filename=document.original_filename,
            file_type=document.file_type.value,
            file_size=document.file_size,
            upload_date=document.created_at.isoformat(),
            status=document.status.value,
            chunk_count=document.chunk_count
        )

        logger.info(f"✅ Document uploaded successfully: {document.id}")
        return document_info
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"❌ Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse, summary="List documents")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> DocumentListResponse:
    """
    List all uploaded documents.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        db: Database session
        
    Returns:
        DocumentListResponse with document list
    """
    try:
        logger.info(f"📋 Listing documents (skip={skip}, limit={limit})")

        # Get documents from manager
        result = await document_manager.list_documents(
            db_session=db,
            skip=skip,
            limit=limit
        )

        # Convert to response format
        documents = []
        for doc_dict in result["documents"]:
            documents.append(DocumentInfo(
                id=doc_dict["id"],
                filename=doc_dict["original_filename"],
                file_type=doc_dict["file_type"],
                file_size=doc_dict["file_size"],
                upload_date=doc_dict["created_at"],
                status=doc_dict["status"],
                chunk_count=doc_dict["chunk_count"]
            ))

        return DocumentListResponse(
            documents=documents,
            total_count=result["total_count"]
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=DocumentInfo, summary="Get document info")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> DocumentInfo:
    """
    Get information about a specific document.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        DocumentInfo for the requested document
    """
    try:
        logger.info(f"📄 Getting document info: {document_id}")

        # Get document from manager
        document = await document_manager.get_document_by_id(document_id, db)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        # Convert to response format
        document_info = DocumentInfo(
            id=document.id,
            filename=document.original_filename,
            file_type=document.file_type.value,
            file_size=document.file_size,
            upload_date=document.created_at.isoformat(),
            status=document.status.value,
            chunk_count=document.chunk_count
        )

        return document_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.delete("/documents/{document_id}", summary="Delete document")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Delete a document and its associated embeddings.
    
    Args:
        document_id: Document ID to delete
        db: Database session
        
    Returns:
        Success confirmation
    """
    try:
        logger.info(f"🗑️ Deleting document: {document_id}")

        # Delete document using manager
        success = await document_manager.delete_document(document_id, db)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        return {"message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Failed to delete document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
