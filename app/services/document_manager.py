"""
BMI Chat Application - Document Manager

This module provides high-level document management operations,
including upload, processing status tracking, and document queries.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from app.models import Document, DocumentChunk, DocumentStatus, DocumentType
from app.services.document_service import DocumentService
from app.core.exceptions import NotFoundError, ValidationError
from loguru import logger


class DocumentManager:
    """
    High-level document management service.
    
    Provides operations for document lifecycle management,
    status tracking, and document queries.
    """
    
    def __init__(self):
        self.document_service = DocumentService()
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        db_session: AsyncSession,
        keywords: Optional[str] = None
    ) -> Document:
        """
        Upload and process a new document.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            db_session: Database session
            keywords: Optional keywords for categorization

        Returns:
            Document: Created document record
        """
        return await self.document_service.upload_document(
            file_content=file_content,
            filename=filename,
            db_session=db_session,
            keywords=keywords
        )
    
    async def get_document_by_id(
        self,
        document_id: str,
        db_session: AsyncSession,
        include_chunks: bool = False
    ) -> Optional[Document]:
        """
        Get document by ID with optional chunk loading.
        
        Args:
            document_id: Document ID
            db_session: Database session
            include_chunks: Whether to load document chunks
            
        Returns:
            Document or None if not found
        """
        try:
            query = select(Document).where(Document.id == document_id)
            
            if include_chunks:
                query = query.options(selectinload(Document.chunks))
            
            result = await db_session.execute(query)
            document = result.scalar_one_or_none()
            
            if document:
                logger.debug(f"📄 Retrieved document: {document_id}")
            
            return document
            
        except Exception as e:
            logger.error(f"❌ Failed to get document {document_id}: {str(e)}")
            return None
    
    async def list_documents(
        self,
        db_session: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[DocumentStatus] = None,
        file_type_filter: Optional[DocumentType] = None,
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List documents with filtering and pagination.
        
        Args:
            db_session: Database session
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            status_filter: Filter by document status
            file_type_filter: Filter by file type
            search_query: Search in filename or title
            
        Returns:
            Dict with documents list and metadata
        """
        try:
            # Build base query
            query = select(Document).where(Document.status != DocumentStatus.DELETED)
            count_query = select(func.count(Document.id)).where(Document.status != DocumentStatus.DELETED)
            
            # Apply filters
            if status_filter:
                query = query.where(Document.status == status_filter)
                count_query = count_query.where(Document.status == status_filter)
            
            if file_type_filter:
                query = query.where(Document.file_type == file_type_filter)
                count_query = count_query.where(Document.file_type == file_type_filter)
            
            if search_query:
                search_filter = or_(
                    Document.original_filename.ilike(f"%{search_query}%"),
                    Document.title.ilike(f"%{search_query}%"),
                    Document.author.ilike(f"%{search_query}%")
                )
                query = query.where(search_filter)
                count_query = count_query.where(search_filter)
            
            # Apply pagination and ordering
            query = query.order_by(Document.created_at.desc()).offset(skip).limit(limit)
            
            # Execute queries
            documents_result = await db_session.execute(query)
            documents = documents_result.scalars().all()
            
            count_result = await db_session.execute(count_query)
            total_count = count_result.scalar()
            
            logger.info(f"📋 Listed {len(documents)} documents (total: {total_count})")
            
            return {
                "documents": [doc.to_dict() for doc in documents],
                "total_count": total_count,
                "skip": skip,
                "limit": limit,
                "has_more": skip + len(documents) < total_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {str(e)}")
            raise
    
    async def delete_document(
        self,
        document_id: str,
        db_session: AsyncSession
    ) -> bool:
        """
        Delete a document and all associated data.
        
        Args:
            document_id: Document ID to delete
            db_session: Database session
            
        Returns:
            bool: True if deleted successfully
        """
        return await self.document_service.delete_document(document_id, db_session)
    
    async def get_document_status(
        self,
        document_id: str,
        db_session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed document processing status.
        
        Args:
            document_id: Document ID
            db_session: Database session
            
        Returns:
            Dict with status information or None if not found
        """
        document = await self.get_document_by_id(document_id, db_session)
        if not document:
            return None
        
        status_info = {
            "id": document.id,
            "filename": document.original_filename,
            "status": document.status.value,
            "file_size": document.file_size,
            "file_type": document.file_type.value,
            "created_at": document.created_at.isoformat(),
            "processing_started_at": document.processing_started_at.isoformat() if document.processing_started_at else None,
            "processing_completed_at": document.processing_completed_at.isoformat() if document.processing_completed_at else None,
            "processing_duration": document.processing_duration,
            "processing_error": document.processing_error,
            "chunk_count": document.chunk_count,
            "vector_count": document.vector_count,
            "query_count": document.query_count,
            "last_queried": document.last_queried.isoformat() if document.last_queried else None
        }
        
        return status_info
    
    async def get_processing_statistics(self, db_session: AsyncSession) -> Dict[str, Any]:
        """
        Get overall document processing statistics.
        
        Args:
            db_session: Database session
            
        Returns:
            Dict with processing statistics
        """
        try:
            # Count documents by status
            status_counts = {}
            for status in DocumentStatus:
                count_query = select(func.count(Document.id)).where(Document.status == status)
                result = await db_session.execute(count_query)
                status_counts[status.value] = result.scalar()
            
            # Get total file size
            size_query = select(func.sum(Document.file_size)).where(Document.status != DocumentStatus.DELETED)
            size_result = await db_session.execute(size_query)
            total_size = size_result.scalar() or 0
            
            # Get total chunks and vectors
            chunk_query = select(func.sum(Document.chunk_count)).where(Document.status == DocumentStatus.PROCESSED)
            chunk_result = await db_session.execute(chunk_query)
            total_chunks = chunk_result.scalar() or 0
            
            vector_query = select(func.sum(Document.vector_count)).where(Document.status == DocumentStatus.PROCESSED)
            vector_result = await db_session.execute(vector_query)
            total_vectors = vector_result.scalar() or 0
            
            # Get query statistics
            query_query = select(func.sum(Document.query_count))
            query_result = await db_session.execute(query_query)
            total_queries = query_result.scalar() or 0
            
            statistics = {
                "status_counts": status_counts,
                "total_documents": sum(status_counts.values()),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_chunks": total_chunks,
                "total_vectors": total_vectors,
                "total_queries": total_queries,
                "average_chunks_per_document": round(total_chunks / max(status_counts.get("processed", 1), 1), 2)
            }
            
            logger.info(f"📊 Retrieved processing statistics")
            return statistics
            
        except Exception as e:
            logger.error(f"❌ Failed to get processing statistics: {str(e)}")
            raise
    
    async def reprocess_document(
        self,
        document_id: str,
        db_session: AsyncSession
    ) -> bool:
        """
        Reprocess a document (useful for failed documents).
        
        Args:
            document_id: Document ID to reprocess
            db_session: Database session
            
        Returns:
            bool: True if reprocessing started successfully
        """
        return await self.document_service.reprocess_document(document_id, db_session)
    
    async def search_documents(
        self,
        query: str,
        db_session: AsyncSession,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by content similarity (placeholder for future implementation).
        
        Args:
            query: Search query
            db_session: Database session
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        # For now, implement simple text search
        # TODO: Implement semantic search using embeddings
        
        search_filter = or_(
            Document.original_filename.ilike(f"%{query}%"),
            Document.title.ilike(f"%{query}%"),
            Document.author.ilike(f"%{query}%"),
            Document.subject.ilike(f"%{query}%")
        )
        
        query_obj = select(Document).where(
            and_(
                Document.status == DocumentStatus.PROCESSED,
                search_filter
            )
        ).order_by(Document.query_count.desc()).limit(limit)
        
        result = await db_session.execute(query_obj)
        documents = result.scalars().all()
        
        return [doc.to_dict() for doc in documents]
