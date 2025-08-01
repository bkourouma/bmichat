"""
BMI Chat Application - Document Processing Service

This module handles document upload, processing, chunking, and embedding generation.
It provides a complete pipeline from raw documents to searchable vector embeddings.
"""

import asyncio
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

import aiofiles
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
)
from langchain.schema import Document as LangChainDocument
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import DocumentProcessingError, ValidationError
from app.models import Document, DocumentChunk, DocumentStatus, DocumentType
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.qa_chunker import QAChunker, ChunkMetadata


class DocumentProcessor:
    """
    Document processing service for handling file uploads and text extraction.
    
    Supports multiple file formats and provides comprehensive error handling.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        self.qa_chunker = QAChunker()
        # Keep fallback splitter for compatibility
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def process_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        db_session: AsyncSession,
        keywords: Optional[str] = None
    ) -> Document:
        """
        Process an uploaded file through the complete pipeline.
        
        Args:
            file_content: Raw file content
            filename: Original filename
            db_session: Database session
            keywords: Optional keywords for categorization

        Returns:
            Document: Processed document record
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        document_id = str(uuid4())
        
        try:
            logger.info(f"📄 Starting document processing: {filename}")
            
            # Validate file
            file_type, mime_type = self._validate_file(filename, file_content)
            
            # Save file to disk
            file_path = await self._save_file(document_id, filename, file_content)
            
            # Create document record
            document = Document(
                id=document_id,
                filename=f"{document_id}_{filename}",
                original_filename=filename,
                file_path=str(file_path),
                file_type=file_type,
                file_size=len(file_content),
                mime_type=mime_type,
                status=DocumentStatus.UPLOADED,
                keywords=keywords  # Add keywords to document
            )
            
            db_session.add(document)
            await db_session.commit()
            
            # Process document asynchronously
            asyncio.create_task(
                self._process_document_async(document_id, db_session)
            )
            
            logger.info(f"✅ Document uploaded successfully: {document_id}")
            return document
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")
    
    def _validate_file(self, filename: str, content: bytes) -> Tuple[DocumentType, str]:
        """
        Validate uploaded file type and content.
        
        Args:
            filename: Original filename
            content: File content
            
        Returns:
            Tuple of (DocumentType, mime_type)
            
        Raises:
            ValidationError: If file is invalid
        """
        if not filename:
            raise ValidationError("Filename is required")
        
        # Get file extension
        file_extension = filename.split('.')[-1].lower()
        
        # Validate file type
        if file_extension not in settings.supported_file_types:
            raise ValidationError(
                f"Unsupported file type: {file_extension}. "
                f"Supported types: {', '.join(settings.supported_file_types)}"
            )
        
        # Map extension to DocumentType
        type_mapping = {
            'pdf': DocumentType.PDF,
            'txt': DocumentType.TXT,
            'docx': DocumentType.DOCX,
            'md': DocumentType.MD
        }
        
        document_type = type_mapping.get(file_extension)
        if not document_type:
            raise ValidationError(f"Unsupported document type: {file_extension}")
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # Validate file size
        max_size_str = settings.max_upload_size
        if max_size_str.endswith('MB'):
            max_size_bytes = int(max_size_str[:-2]) * 1024 * 1024
        elif max_size_str.endswith('KB'):
            max_size_bytes = int(max_size_str[:-2]) * 1024
        else:
            max_size_bytes = int(max_size_str)
        
        if len(content) > max_size_bytes:
            raise ValidationError(
                f"File size ({len(content)} bytes) exceeds maximum allowed size ({max_size_str})"
            )
        
        # Basic content validation
        if len(content) == 0:
            raise ValidationError("File is empty")
        
        return document_type, mime_type
    
    async def _save_file(self, document_id: str, filename: str, content: bytes) -> Path:
        """
        Save uploaded file to disk.
        
        Args:
            document_id: Unique document ID
            filename: Original filename
            content: File content
            
        Returns:
            Path: Saved file path
        """
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_filename = f"{document_id}_{filename}"
        file_path = upload_dir / safe_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        logger.debug(f"💾 File saved: {file_path}")
        return file_path
    
    async def _process_document_async(self, document_id: str, db_session: AsyncSession):
        """
        Process document asynchronously (extract text, chunk, embed).
        
        Args:
            document_id: Document ID to process
            db_session: Database session
        """
        try:
            # Get document from database
            document = await db_session.get(Document, document_id)
            if not document:
                raise DocumentProcessingError(f"Document not found: {document_id}")
            
            # Mark processing started
            document.mark_processing_started()
            await db_session.commit()
            
            logger.info(f"🔄 Processing document: {document.original_filename}")
            
            # Extract text
            text_content, metadata = await self._extract_text(document)
            
            # Update document with metadata
            self._update_document_metadata(document, text_content, metadata)
            
            # Create chunks
            chunks = await self._create_chunks(document, text_content)
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(chunks)
            
            # Store in vector database
            await self._store_vectors(document, chunks, embeddings)
            
            # Save chunks to database
            await self._save_chunks(document, chunks, db_session)
            
            # Mark processing completed
            document.mark_processing_completed(
                chunk_count=len(chunks),
                vector_count=len(embeddings)
            )
            
            await db_session.commit()
            
            logger.info(f"✅ Document processing completed: {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            
            # Mark processing failed
            if document:
                document.mark_processing_failed(str(e))
                await db_session.commit()
    
    async def _extract_text(self, document: Document) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text content from document file.
        
        Args:
            document: Document record
            
        Returns:
            Tuple of (text_content, metadata)
        """
        file_path = Path(document.file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        try:
            if document.file_type == DocumentType.PDF:
                return await self._extract_pdf_text(file_path)
            elif document.file_type == DocumentType.TXT:
                return await self._extract_txt_text(file_path)
            elif document.file_type == DocumentType.DOCX:
                return await self._extract_docx_text(file_path)
            elif document.file_type == DocumentType.MD:
                return await self._extract_md_text(file_path)
            else:
                raise DocumentProcessingError(f"Unsupported file type: {document.file_type}")
                
        except Exception as e:
            raise DocumentProcessingError(f"Text extraction failed: {str(e)}")
    
    async def _extract_pdf_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF file."""
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        metadata = {
            "total_pages": len(documents),
            "total_characters": len(text_content),
            "total_words": len(text_content.split()),
        }
        
        # Extract PDF metadata if available
        if documents and documents[0].metadata:
            pdf_metadata = documents[0].metadata
            metadata.update({
                "title": pdf_metadata.get("title"),
                "author": pdf_metadata.get("author"),
                "subject": pdf_metadata.get("subject"),
            })
        
        return text_content, metadata
    
    async def _extract_txt_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from TXT file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            text_content = await f.read()
        
        metadata = {
            "total_pages": 1,
            "total_characters": len(text_content),
            "total_words": len(text_content.split()),
        }
        
        return text_content, metadata
    
    async def _extract_docx_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from DOCX file."""
        loader = UnstructuredWordDocumentLoader(str(file_path))
        documents = loader.load()
        
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        metadata = {
            "total_pages": 1,  # DOCX doesn't have clear page breaks
            "total_characters": len(text_content),
            "total_words": len(text_content.split()),
        }
        
        return text_content, metadata
    
    async def _extract_md_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from Markdown file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            text_content = await f.read()
        
        metadata = {
            "total_pages": 1,
            "total_characters": len(text_content),
            "total_words": len(text_content.split()),
        }
        
        return text_content, metadata

    def _update_document_metadata(self, document: Document, text_content: str, metadata: Dict[str, Any]):
        """Update document with extracted metadata."""
        document.total_pages = metadata.get("total_pages")
        document.total_words = metadata.get("total_words")
        document.total_characters = metadata.get("total_characters")
        document.title = metadata.get("title")
        document.author = metadata.get("author")
        document.subject = metadata.get("subject")

    async def _create_chunks(self, document: Document, text_content: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Create text chunks from document content using QA-aware chunking."""
        if not text_content.strip():
            raise DocumentProcessingError("Document contains no text content")

        # Prepare document metadata for chunking
        doc_metadata = {
            "source": document.original_filename,
            "document_id": document.id,
            "file_type": document.file_type.value,
            "keywords": document.keywords
        }

        # Use QA-aware chunker
        chunk_tuples = self.qa_chunker.chunk_document(text_content, doc_metadata)

        # Convert to format expected by the rest of the pipeline
        chunks_with_metadata = []
        for chunk_text, chunk_meta in chunk_tuples:
            # Combine chunk metadata with document metadata
            combined_metadata = {
                **doc_metadata,
                "chunk_type": chunk_meta.chunk_type.value,
                "chunk_length": chunk_meta.length,
                "word_count": chunk_meta.word_count,
                "has_questions": chunk_meta.has_questions,
                "has_answers": chunk_meta.has_answers,
                "confidence_score": chunk_meta.confidence_score
            }
            chunks_with_metadata.append((chunk_text, combined_metadata))

        # Limit number of chunks
        if len(chunks_with_metadata) > settings.max_chunks_per_document:
            logger.warning(
                f"Document has {len(chunks_with_metadata)} chunks, limiting to {settings.max_chunks_per_document}"
            )
            chunks_with_metadata = chunks_with_metadata[:settings.max_chunks_per_document]

        # Get chunking summary
        summary = self.qa_chunker.get_chunk_summary(chunk_tuples)
        logger.info(f"📝 Created {len(chunks_with_metadata)} chunks for document {document.id}")
        logger.info(f"📊 Chunking summary: {summary}")

        return chunks_with_metadata

    async def _generate_embeddings(self, chunks_with_metadata: List[Tuple[str, Dict[str, Any]]]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        chunk_texts = [chunk_text for chunk_text, _ in chunks_with_metadata]
        logger.info(f"🧠 Generating embeddings for {len(chunk_texts)} chunks")

        embeddings = await self.embedding_service.generate_embeddings(chunk_texts)

        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings

    async def _store_vectors(
        self,
        document: Document,
        chunks_with_metadata: List[Tuple[str, Dict[str, Any]]],
        embeddings: List[List[float]]
    ):
        """Store vectors in ChromaDB with enhanced metadata."""
        logger.info(f"💾 Storing {len(embeddings)} vectors in ChromaDB")

        # Extract chunks and prepare enhanced metadata
        chunks = []
        metadata_list = []

        for i, (chunk_text, chunk_metadata) in enumerate(chunks_with_metadata):
            chunks.append(chunk_text)

            # Enhanced metadata including QA information and keywords
            enhanced_metadata = {
                "document_id": document.id,
                "chunk_index": i,
                "filename": document.original_filename,
                "file_type": document.file_type.value,
                "chunk_length": len(chunk_text),
                "created_at": datetime.utcnow().isoformat(),
                "keywords": document.keywords or "",
                # QA-specific metadata
                "chunk_type": chunk_metadata.get("chunk_type", "regular"),
                "has_questions": chunk_metadata.get("has_questions", False),
                "has_answers": chunk_metadata.get("has_answers", False),
                "confidence_score": chunk_metadata.get("confidence_score", 1.0),
                "word_count": chunk_metadata.get("word_count", 0)
            }
            metadata_list.append(enhanced_metadata)

        # Store in vector database
        await self.vector_service.add_document_chunks(
            document_id=document.id,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata_list
        )

        logger.info(f"✅ Stored vectors for document {document.id}")

    async def _save_chunks(
        self,
        document: Document,
        chunks_with_metadata: List[Tuple[str, Dict[str, Any]]],
        db_session: AsyncSession
    ):
        """Save chunks to database with enhanced metadata."""
        logger.info(f"💾 Saving {len(chunks_with_metadata)} chunks to database")

        for i, (chunk_text, chunk_metadata) in enumerate(chunks_with_metadata):
            chunk = DocumentChunk(
                id=str(uuid4()),
                document_id=document.id,
                chunk_index=i,
                content=chunk_text,
                content_length=len(chunk_text),
                vector_id=f"{document.id}_chunk_{i}",
                embedding_model="text-embedding-ada-002"
            )
            db_session.add(chunk)

        await db_session.commit()
        logger.info(f"✅ Saved chunks for document {document.id}")


class DocumentService:
    """
    High-level document service for managing document operations.

    Provides a clean interface for document upload, processing, and management.
    """

    def __init__(self):
        self.processor = DocumentProcessor()

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
        return await self.processor.process_uploaded_file(
            file_content=file_content,
            filename=filename,
            db_session=db_session,
            keywords=keywords
        )

    async def get_document(self, document_id: str, db_session: AsyncSession) -> Optional[Document]:
        """Get document by ID."""
        return await db_session.get(Document, document_id)

    async def delete_document(self, document_id: str, db_session: AsyncSession) -> bool:
        """
        Delete document and all associated data.

        Args:
            document_id: Document ID to delete
            db_session: Database session

        Returns:
            bool: True if deleted successfully
        """
        try:
            document = await db_session.get(Document, document_id)
            if not document:
                return False

            # Delete from vector database
            await self.processor.vector_service.delete_document(document_id)

            # Delete file from disk
            file_path = Path(document.file_path)
            if file_path.exists():
                file_path.unlink()

            # Mark as deleted in database
            document.status = DocumentStatus.DELETED
            await db_session.commit()

            logger.info(f"✅ Document deleted: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {str(e)}")
            return False

    async def reprocess_document(self, document_id: str, db_session: AsyncSession) -> bool:
        """
        Reprocess an existing document.

        Args:
            document_id: Document ID to reprocess
            db_session: Database session

        Returns:
            bool: True if reprocessing started successfully
        """
        try:
            document = await db_session.get(Document, document_id)
            if not document:
                return False

            # Reset status
            document.status = DocumentStatus.UPLOADED
            document.processing_error = None
            await db_session.commit()

            # Start reprocessing
            asyncio.create_task(
                self.processor._process_document_async(document_id, db_session)
            )

            logger.info(f"🔄 Document reprocessing started: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start reprocessing for {document_id}: {str(e)}")
            return False
