"""
BMI Chat Application - Chat Service Tests

Test suite for chat service and RAG functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from app.services.chat_service import ChatService
from app.models import User, ChatSession, ChatMessage, MessageRole, ChatSessionStatus
from app.core.exceptions import ChatError, ValidationError


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.get = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service."""
    service = AsyncMock()
    service.hybrid_retrieve = AsyncMock()
    return service


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = AsyncMock()
    
    # Mock response structure
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Bonjour! Je suis Akissi, votre assistante BMI."
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.total_tokens = 150
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


@pytest.fixture
def chat_service(mock_retrieval_service, mock_openai_client):
    """Create chat service with mocked dependencies."""
    service = ChatService()
    service.retrieval_service = mock_retrieval_service
    service.openai_client = mock_openai_client
    return service


@pytest.fixture
def sample_context_chunks():
    """Sample context chunks for testing."""
    return [
        {
            "content": "Q: Qu'est-ce que l'assurance BMI? R: L'assurance BMI offre une couverture complète pour vos besoins de santé.",
            "score": 0.85,
            "distance": 0.15,
            "chunk_type": "qa_pair",
            "has_questions": True,
            "has_answers": True,
            "keywords": "assurance,santé,couverture",
            "document_info": {
                "id": "doc1",
                "filename": "guide_assurance.pdf",
                "title": "Guide Assurance BMI"
            },
            "metadata": {
                "chunk_index": 0,
                "document_id": "doc1"
            }
        },
        {
            "content": "Nos plans d'assurance incluent la couverture médicale, dentaire et visuelle.",
            "score": 0.72,
            "distance": 0.28,
            "chunk_type": "regular",
            "has_questions": False,
            "has_answers": False,
            "keywords": "assurance,médical,dentaire",
            "document_info": {
                "id": "doc1",
                "filename": "guide_assurance.pdf",
                "title": "Guide Assurance BMI"
            },
            "metadata": {
                "chunk_index": 1,
                "document_id": "doc1"
            }
        }
    ]


class TestChatService:
    """Test chat service functionality."""
    
    @pytest.mark.asyncio
    async def test_process_chat_message_success(
        self, 
        chat_service, 
        mock_db_session, 
        sample_context_chunks
    ):
        """Test successful chat message processing."""
        # Setup mocks
        chat_service.retrieval_service.hybrid_retrieve.return_value = sample_context_chunks
        
        # Mock user and session creation
        mock_user = User(id="user1", session_id="session1", preferred_language="fr")
        mock_session = ChatSession(
            id="session1", 
            user_id="user1", 
            status=ChatSessionStatus.ACTIVE,
            message_count=0
        )
        
        mock_db_session.get.side_effect = [None, None]  # User and session don't exist
        
        # Process message
        result = await chat_service.process_chat_message(
            message="Qu'est-ce que l'assurance BMI?",
            session_id="session1",
            db_session=mock_db_session
        )
        
        # Verify result structure
        assert "message" in result
        assert "session_id" in result
        assert "message_id" in result
        assert "sources" in result
        assert "timestamp" in result
        assert "response_time_ms" in result
        assert "tokens_used" in result
        assert "context_used" in result
        
        # Verify session ID
        assert result["session_id"] == "session1"
        
        # Verify context was used
        assert result["context_used"] == len(sample_context_chunks)
        
        # Verify sources
        assert len(result["sources"]) == len(sample_context_chunks)
        assert result["sources"][0]["document_id"] == "doc1"
        assert result["sources"][0]["filename"] == "guide_assurance.pdf"
    
    @pytest.mark.asyncio
    async def test_process_chat_message_empty_message(
        self, 
        chat_service, 
        mock_db_session
    ):
        """Test processing empty message raises validation error."""
        with pytest.raises(ValidationError):
            await chat_service.process_chat_message(
                message="",
                session_id="session1",
                db_session=mock_db_session
            )
    
    @pytest.mark.asyncio
    async def test_build_context_text(self, chat_service, sample_context_chunks):
        """Test context text building."""
        context_text = chat_service._build_context_text(sample_context_chunks)
        
        assert "Source 1: guide_assurance.pdf - Q&A" in context_text
        assert "Source 2: guide_assurance.pdf" in context_text
        assert "Qu'est-ce que l'assurance BMI?" in context_text
        assert "couverture médicale, dentaire" in context_text
    
    @pytest.mark.asyncio
    async def test_build_system_prompt(self, chat_service):
        """Test system prompt building."""
        context_text = "Test context about BMI insurance"
        prompt = chat_service._build_system_prompt(context_text)
        
        assert "Tu es Akissi" in prompt
        assert "BMI (Bureau Médical International)" in prompt
        assert "français" in prompt
        assert "Test context about BMI insurance" in prompt
        assert "CONTEXTE DOCUMENTAIRE:" in prompt
    
    @pytest.mark.asyncio
    async def test_build_conversation_messages(self, chat_service):
        """Test conversation messages building."""
        system_prompt = "Tu es Akissi, l'assistante BMI."
        chat_history = [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"}
        ]
        user_message = "Qu'est-ce que BMI?"
        
        messages = chat_service._build_conversation_messages(
            system_prompt, chat_history, user_message
        )
        
        assert len(messages) >= 3  # System + history + current message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == user_message
    
    @pytest.mark.asyncio
    async def test_calculate_cost(self, chat_service):
        """Test cost calculation."""
        # Test with 1000 tokens
        cost = chat_service._calculate_cost(1000)
        assert cost == 0.01  # $0.01 per 1K tokens
        
        # Test with 2500 tokens
        cost = chat_service._calculate_cost(2500)
        assert cost == 0.025  # $0.025 for 2.5K tokens
    
    @pytest.mark.asyncio
    async def test_get_session_summary_existing(self, chat_service, mock_db_session):
        """Test getting session summary for existing session."""
        # Mock existing session and user
        mock_session = ChatSession(
            id="session1",
            user_id="user1",
            status=ChatSessionStatus.ACTIVE,
            message_count=5,
            total_tokens_used=500,
            total_cost=0.005
        )
        mock_user = User(
            id="user1",
            session_id="session1",
            preferred_language="fr",
            total_sessions=2,
            total_messages=10
        )
        
        mock_db_session.get.side_effect = [mock_session, mock_user]
        
        summary = await chat_service.get_session_summary("session1", mock_db_session)
        
        assert summary is not None
        assert summary["session_id"] == "session1"
        assert summary["status"] == "active"
        assert summary["message_count"] == 5
        assert summary["total_tokens_used"] == 500
        assert summary["total_cost"] == 0.005
        assert summary["user_info"]["preferred_language"] == "fr"
        assert summary["user_info"]["total_sessions"] == 2
    
    @pytest.mark.asyncio
    async def test_get_session_summary_nonexistent(self, chat_service, mock_db_session):
        """Test getting session summary for non-existent session."""
        mock_db_session.get.return_value = None
        
        summary = await chat_service.get_session_summary("nonexistent", mock_db_session)
        
        assert summary is None
    
    @pytest.mark.asyncio
    async def test_end_session_success(self, chat_service, mock_db_session):
        """Test ending session successfully."""
        mock_session = ChatSession(
            id="session1",
            user_id="user1",
            status=ChatSessionStatus.ACTIVE
        )
        mock_db_session.get.return_value = mock_session
        
        result = await chat_service.end_session("session1", mock_db_session)
        
        assert result is True
        assert mock_session.status == ChatSessionStatus.INACTIVE
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_end_session_not_found(self, chat_service, mock_db_session):
        """Test ending non-existent session."""
        mock_db_session.get.return_value = None
        
        result = await chat_service.end_session("nonexistent", mock_db_session)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_ai_response_with_context(
        self, 
        chat_service, 
        sample_context_chunks
    ):
        """Test AI response generation with context."""
        user_message = "Qu'est-ce que l'assurance BMI?"
        chat_history = []
        
        response, metadata = await chat_service._generate_ai_response(
            user_message, sample_context_chunks, chat_history
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        assert "model" in metadata
        assert "tokens_used" in metadata
        assert "response_time_ms" in metadata
        assert "context_chunks_used" in metadata
        assert metadata["context_chunks_used"] == len(sample_context_chunks)
    
    @pytest.mark.asyncio
    async def test_generate_ai_response_no_context(self, chat_service):
        """Test AI response generation without context."""
        user_message = "Bonjour"
        chat_history = []
        context_chunks = []
        
        response, metadata = await chat_service._generate_ai_response(
            user_message, context_chunks, chat_history
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert metadata["context_chunks_used"] == 0
    
    @pytest.mark.asyncio
    async def test_retrieval_service_integration(
        self, 
        chat_service, 
        mock_db_session, 
        sample_context_chunks
    ):
        """Test integration with retrieval service."""
        # Setup retrieval service mock
        chat_service.retrieval_service.hybrid_retrieve.return_value = sample_context_chunks
        
        # Test context retrieval
        context = await chat_service._retrieve_context(
            message="assurance BMI",
            keywords_filter=["assurance"],
            max_chunks=3,
            db_session=mock_db_session
        )
        
        assert len(context) == len(sample_context_chunks)
        assert context[0]["content"] == sample_context_chunks[0]["content"]
        
        # Verify retrieval service was called correctly
        chat_service.retrieval_service.hybrid_retrieve.assert_called_once_with(
            query="assurance BMI",
            keywords=["assurance"],
            db_session=mock_db_session,
            k=3,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
