"""
BMI Chat Application - Chat Router

This module provides chat endpoints for the BMI Chat application.
Handles chat requests, message processing, and response generation.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from loguru import logger

from app.config import settings
from app.core.database import get_db
from app.core.exceptions import ChatError, ValidationError
from app.services.chat_service import ChatService


router = APIRouter()

# Initialize chat service
chat_service = ChatService()


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    user_id: Optional[str] = Field(None, description="User ID")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords filter for context")
    use_history: Optional[bool] = Field(default=True, description="Include chat history")
    max_context_chunks: Optional[int] = Field(default=None, description="Max context chunks")


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message ID")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents with metadata")
    timestamp: str = Field(..., description="Response timestamp")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    tokens_used: int = Field(..., description="Tokens used in response")
    context_used: int = Field(..., description="Number of context chunks used")
    has_history: bool = Field(..., description="Whether chat history was used")


@router.post("/chat", response_model=ChatResponse, summary="Send chat message")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
) -> ChatResponse:
    """
    Process a chat message and return AI response.
    
    Args:
        request: Chat request containing message and context
        db: Database session
        
    Returns:
        ChatResponse with AI-generated response
        
    Raises:
        HTTPException: If chat processing fails
    """
    try:
        logger.info(f"💬 Processing chat message: {request.message[:50]}...")

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid4())

        # Process chat message using chat service
        response_data = await chat_service.process_chat_message(
            message=request.message,
            session_id=session_id,
            db_session=db,
            user_id=request.user_id,
            keywords_filter=request.keywords,
            use_history=request.use_history or True,
            max_context_chunks=request.max_context_chunks
        )

        # Convert to response format
        response = ChatResponse(
            message=response_data["message"],
            session_id=response_data["session_id"],
            message_id=response_data["message_id"],
            sources=response_data["sources"],
            timestamp=response_data["timestamp"],
            response_time_ms=response_data["response_time_ms"],
            tokens_used=response_data["tokens_used"],
            context_used=response_data["context_used"],
            has_history=response_data["has_history"]
        )

        logger.info(f"✅ Chat response generated for session {session_id}")
        return response
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.get("/chat/sessions/{session_id}/history", summary="Get chat history")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
) -> List[ChatMessage]:
    """
    Retrieve chat history for a session.
    
    Args:
        session_id: Chat session ID
        limit: Maximum number of messages to return
        db: Database session
        
    Returns:
        List of chat messages
    """
    try:
        logger.info(f"📜 Retrieving chat history for session {session_id}")

        # Get session summary which includes message count
        summary = await chat_service.get_session_summary(session_id, db)

        if not summary:
            return []

        # Get recent messages from database
        from sqlalchemy import select
        from app.models import ChatMessage

        query = select(ChatMessage).where(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.message_index.desc()).limit(limit * 2)

        result = await db.execute(query)
        messages = result.scalars().all()

        # Convert to response format
        history = []
        for message in reversed(messages):
            history.append(ChatMessage(
                role=message.role.value,
                content=message.content,
                timestamp=message.timestamp.isoformat() if message.timestamp else None
            ))

        return history
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )


@router.delete("/chat/sessions/{session_id}", summary="Clear chat session")
async def clear_chat_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Clear a chat session and its history.
    
    Args:
        session_id: Chat session ID to clear
        db: Database session
        
    Returns:
        Success confirmation
    """
    try:
        logger.info(f"🗑️ Clearing chat session {session_id}")

        # End the session using chat service
        success = await chat_service.end_session(session_id, db)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"❌ Failed to clear chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear chat session: {str(e)}"
        )


@router.get("/chat/sessions/{session_id}/summary", summary="Get session summary")
async def get_session_summary(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get session summary and statistics.

    Args:
        session_id: Chat session ID
        db: Database session

    Returns:
        Session summary with statistics
    """
    try:
        logger.info(f"📊 Getting session summary: {session_id}")

        summary = await chat_service.get_session_summary(session_id, db)

        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get session summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session summary: {str(e)}"
        )
