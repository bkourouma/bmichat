"""
BMI Chat Application - Search Router

This module provides search endpoints for testing and debugging
the vector database and retrieval functionality.
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from loguru import logger

from app.config import settings
from app.core.database import get_db
from app.core.exceptions import VectorDatabaseError
from app.services.retrieval_service import RetrievalService
from app.services.vector_service import VectorService


router = APIRouter()

# Initialize services
retrieval_service = RetrievalService()
vector_service = VectorService()


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of results")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords filter")
    document_ids: Optional[List[str]] = Field(default=None, description="Document IDs filter")
    chunk_type: Optional[str] = Field(default=None, description="Chunk type filter")
    min_score: Optional[float] = Field(default=0.3, ge=0.0, le=1.0, description="Minimum relevance score")


class HybridSearchRequest(BaseModel):
    """Hybrid search request model."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords for hybrid search")
    k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of results")
    semantic_weight: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Semantic similarity weight")
    keyword_weight: Optional[float] = Field(default=0.3, ge=0.0, le=1.0, description="Keyword matching weight")


class SearchResult(BaseModel):
    """Search result model."""
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    distance: float = Field(..., description="Vector distance")
    chunk_type: str = Field(..., description="Type of chunk")
    has_questions: bool = Field(..., description="Contains questions")
    has_answers: bool = Field(..., description="Contains answers")
    keywords: str = Field(..., description="Document keywords")
    document_info: Optional[Dict[str, Any]] = Field(None, description="Document information")
    relevance_explanation: str = Field(..., description="Why this result is relevant")


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total results found")
    query: str = Field(..., description="Original query")
    search_time_ms: Optional[float] = Field(None, description="Search time in milliseconds")


@router.post("/search/semantic", response_model=SearchResponse, summary="Semantic search")
async def semantic_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
) -> SearchResponse:
    """
    Perform semantic similarity search on document chunks.
    
    Args:
        request: Search request parameters
        db: Database session
        
    Returns:
        SearchResponse with relevant chunks
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"🔍 Semantic search: {request.query[:50]}...")
        
        # Perform retrieval
        chunks = await retrieval_service.retrieve_relevant_chunks(
            query=request.query,
            db_session=db,
            k=request.k,
            keywords_filter=request.keywords,
            document_ids_filter=request.document_ids,
            prefer_qa_pairs=True,
            min_relevance_score=request.min_score or 0.3
        )
        
        # Convert to response format
        results = []
        for chunk in chunks:
            results.append(SearchResult(
                content=chunk["content"],
                score=chunk["score"],
                distance=chunk["distance"],
                chunk_type=chunk["chunk_type"],
                has_questions=chunk["has_questions"],
                has_answers=chunk["has_answers"],
                keywords=chunk["keywords"],
                document_info=chunk["document_info"],
                relevance_explanation=chunk["relevance_explanation"]
            ))
        
        search_time = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            results=results,
            total_found=len(results),
            query=request.query,
            search_time_ms=round(search_time, 2)
        )
        
        logger.info(f"✅ Semantic search completed: {len(results)} results in {search_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Semantic search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/search/keywords", response_model=SearchResponse, summary="Keyword search")
async def keyword_search(
    keywords: List[str] = Query(..., description="Keywords to search for"),
    k: Optional[int] = Query(default=None, ge=1, le=20, description="Number of results"),
    chunk_type: Optional[str] = Query(default=None, description="Chunk type filter"),
    db: AsyncSession = Depends(get_db)
) -> SearchResponse:
    """
    Perform keyword-based search on document chunks.
    
    Args:
        keywords: List of keywords to search for
        k: Number of results to return
        chunk_type: Optional chunk type filter
        db: Database session
        
    Returns:
        SearchResponse with matching chunks
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"🏷️ Keyword search: {keywords}")
        
        # Perform keyword retrieval
        chunks = await retrieval_service.retrieve_by_keywords(
            keywords=keywords,
            db_session=db,
            k=k,
            chunk_type_filter=chunk_type
        )
        
        # Convert to response format
        results = []
        for chunk in chunks:
            results.append(SearchResult(
                content=chunk["content"],
                score=chunk["score"],
                distance=chunk["distance"],
                chunk_type=chunk["chunk_type"],
                has_questions=chunk["has_questions"],
                has_answers=chunk["has_answers"],
                keywords=chunk["keywords"],
                document_info=chunk["document_info"],
                relevance_explanation=chunk["relevance_explanation"]
            ))
        
        search_time = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            results=results,
            total_found=len(results),
            query=f"keywords: {', '.join(keywords)}",
            search_time_ms=round(search_time, 2)
        )
        
        logger.info(f"✅ Keyword search completed: {len(results)} results in {search_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Keyword search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Keyword search failed: {str(e)}"
        )


@router.post("/search/hybrid", response_model=SearchResponse, summary="Hybrid search")
async def hybrid_search(
    request: HybridSearchRequest,
    db: AsyncSession = Depends(get_db)
) -> SearchResponse:
    """
    Perform hybrid search combining semantic similarity and keyword matching.
    
    Args:
        request: Hybrid search request parameters
        db: Database session
        
    Returns:
        SearchResponse with hybrid results
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"🔄 Hybrid search: {request.query[:50]}...")
        
        # Perform hybrid retrieval
        chunks = await retrieval_service.hybrid_retrieve(
            query=request.query,
            keywords=request.keywords,
            db_session=db,
            k=request.k,
            semantic_weight=request.semantic_weight or 0.7,
            keyword_weight=request.keyword_weight or 0.3
        )
        
        # Convert to response format
        results = []
        for chunk in chunks:
            results.append(SearchResult(
                content=chunk["content"],
                score=chunk["score"],
                distance=chunk["distance"],
                chunk_type=chunk["chunk_type"],
                has_questions=chunk["has_questions"],
                has_answers=chunk["has_answers"],
                keywords=chunk["keywords"],
                document_info=chunk["document_info"],
                relevance_explanation=chunk["relevance_explanation"]
            ))
        
        search_time = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            results=results,
            total_found=len(results),
            query=request.query,
            search_time_ms=round(search_time, 2)
        )
        
        logger.info(f"✅ Hybrid search completed: {len(results)} results in {search_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Hybrid search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}"
        )


@router.get("/search/analytics", summary="Get search analytics")
async def get_search_analytics(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get search and retrieval analytics.
    
    Args:
        days: Number of days to analyze
        db: Database session
        
    Returns:
        Dict with analytics data
    """
    try:
        logger.info(f"📊 Getting search analytics for {days} days")
        
        analytics = await retrieval_service.get_retrieval_analytics(
            db_session=db,
            days=days
        )
        
        logger.info("✅ Search analytics retrieved")
        return analytics
        
    except Exception as e:
        logger.error(f"❌ Failed to get search analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/search/stats", summary="Get vector database statistics")
async def get_vector_stats() -> Dict[str, Any]:
    """
    Get vector database statistics.
    
    Returns:
        Dict with vector database stats
    """
    try:
        logger.info("📊 Getting vector database statistics")
        
        stats = await vector_service.get_collection_stats()
        
        logger.info("✅ Vector database stats retrieved")
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get vector stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector stats: {str(e)}"
        )
