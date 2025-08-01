"""
BMI Chat Application - Chat Service

This module provides the core chat functionality with RAG (Retrieval Augmented Generation).
Combines document retrieval with OpenAI GPT-4o for intelligent, context-aware responses.
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from openai import AsyncOpenAI
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.exceptions import ChatError, OpenAIError, ValidationError
from app.models import User, ChatSession, ChatMessage, MessageRole, ChatSessionStatus
from app.services.retrieval_service import RetrievalService


class ChatService:
    """
    Core chat service with RAG (Retrieval Augmented Generation).
    
    Provides intelligent responses by combining document retrieval
    with OpenAI GPT-4o for context-aware conversations.
    """
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.retrieval_service = RetrievalService()
        self.model = settings.openai_model  # GPT-4o
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
    
    async def process_chat_message(
        self,
        message: str,
        session_id: str,
        db_session: AsyncSession,
        user_id: Optional[str] = None,
        keywords_filter: Optional[List[str]] = None,
        use_history: bool = True,
        max_context_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate an AI response.
        
        Args:
            message: User message text
            session_id: Chat session ID
            db_session: Database session
            user_id: Optional user ID
            keywords_filter: Optional keywords for document filtering
            use_history: Whether to include chat history in context
            max_context_chunks: Maximum number of context chunks to retrieve
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"💬 Processing chat message: {message[:50]}...")
            
            # Validate input
            if not message.strip():
                raise ValidationError("Message cannot be empty")
            
            # Get or create user and session
            user, chat_session = await self._get_or_create_session(
                session_id, user_id, db_session
            )
            
            # Save user message
            user_message = await self._save_user_message(
                message, chat_session, db_session
            )
            
            # Retrieve relevant context
            context_chunks = await self._retrieve_context(
                message, keywords_filter, max_context_chunks, db_session
            )
            
            # Get chat history if requested
            chat_history = []
            if use_history:
                chat_history = await self._get_chat_history(chat_session, db_session)
            
            # Generate AI response
            ai_response, response_metadata = await self._generate_ai_response(
                message, context_chunks, chat_history
            )
            
            # Save AI response
            ai_message = await self._save_ai_message(
                ai_response, chat_session, response_metadata, context_chunks, db_session
            )
            
            # Update session statistics
            await self._update_session_stats(
                chat_session, response_metadata, db_session
            )
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Prepare response
            response = {
                "message": ai_response,
                "session_id": session_id,
                "message_id": ai_message.id,
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": round(response_time, 2),
                "tokens_used": response_metadata.get("tokens_used", 0),
                "sources": [
                    {
                        "document_id": chunk.get("document_info", {}).get("id"),
                        "filename": chunk.get("document_info", {}).get("filename"),
                        "chunk_type": chunk.get("chunk_type"),
                        "relevance_score": chunk.get("score", 0.0)
                    }
                    for chunk in context_chunks
                ],
                "context_used": len(context_chunks),
                "has_history": len(chat_history) > 0
            }
            
            logger.info(f"✅ Chat response generated in {response_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"❌ Chat processing failed: {str(e)}")
            
            # Try to save error message
            try:
                if 'chat_session' in locals():
                    await self._save_error_message(str(e), chat_session, db_session)
            except:
                pass
            
            raise ChatError(f"Failed to process chat message: {str(e)}")
    
    async def _get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str],
        db_session: AsyncSession
    ) -> Tuple[User, ChatSession]:
        """Get or create user and chat session."""
        from sqlalchemy import select

        # Get or create user
        if user_id:
            # Try to get user by ID first
            user = await db_session.get(User, user_id)
            if not user:
                # Check if user exists by session_id
                query = select(User).where(User.session_id == session_id)
                result = await db_session.execute(query)
                user = result.scalar_one_or_none()

                if not user:
                    user = User(
                        id=user_id,
                        session_id=session_id,
                        preferred_language="fr"
                    )
                    db_session.add(user)
        else:
            # Try to get existing user by session_id first
            query = select(User).where(User.session_id == session_id)
            result = await db_session.execute(query)
            user = result.scalar_one_or_none()

            if not user:
                # Create anonymous user
                user = User(
                    id=str(uuid4()),
                    session_id=session_id,
                    preferred_language="fr"
                )
                db_session.add(user)

        # Get or create chat session
        chat_session = await db_session.get(ChatSession, session_id)
        if not chat_session:
            chat_session = ChatSession(
                id=session_id,
                user_id=user.id,
                model_name=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                status=ChatSessionStatus.ACTIVE
            )
            db_session.add(chat_session)

        await db_session.commit()
        return user, chat_session
    
    async def _save_user_message(
        self,
        message: str,
        chat_session: ChatSession,
        db_session: AsyncSession
    ) -> ChatMessage:
        """Save user message to database."""
        user_message = ChatMessage(
            id=str(uuid4()),
            session_id=chat_session.id,
            role=MessageRole.USER,
            content=message,
            message_index=chat_session.message_count,
            timestamp=datetime.utcnow()
        )
        
        db_session.add(user_message)
        chat_session.add_message_stats()
        await db_session.commit()
        
        return user_message
    
    async def _retrieve_context(
        self,
        message: str,
        keywords_filter: Optional[List[str]],
        max_chunks: Optional[int],
        db_session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the message."""
        max_chunks = max_chunks or settings.default_retrieval_k
        
        try:
            # Use hybrid retrieval for best results
            context_chunks = await self.retrieval_service.hybrid_retrieve(
                query=message,
                keywords=keywords_filter,
                db_session=db_session,
                k=max_chunks,
                semantic_weight=0.7,
                keyword_weight=0.3
            )
            
            logger.info(f"📚 Retrieved {len(context_chunks)} context chunks")
            return context_chunks
            
        except Exception as e:
            logger.warning(f"⚠️ Context retrieval failed: {str(e)}")
            return []
    
    async def _get_chat_history(
        self,
        chat_session: ChatSession,
        db_session: AsyncSession,
        limit: int = None
    ) -> List[Dict[str, str]]:
        """Get recent chat history for context."""
        limit = limit or settings.max_chat_history
        
        try:
            from sqlalchemy import select
            
            # Get recent messages
            query = select(ChatMessage).where(
                ChatMessage.session_id == chat_session.id
            ).order_by(ChatMessage.message_index.desc()).limit(limit * 2)  # Get more for user/assistant pairs
            
            result = await db_session.execute(query)
            messages = result.scalars().all()
            
            # Convert to chat format (reverse to chronological order)
            history = []
            for message in reversed(messages):
                history.append({
                    "role": message.role.value,
                    "content": message.content
                })
            
            # Limit to specified number of exchanges
            if len(history) > limit * 2:
                history = history[-(limit * 2):]
            
            logger.debug(f"📜 Retrieved {len(history)} history messages")
            return history
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get chat history: {str(e)}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_ai_response(
        self,
        user_message: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate AI response using OpenAI GPT-4o with RAG."""
        try:
            # Build context from retrieved chunks
            context_text = self._build_context_text(context_chunks)

            # Build system prompt
            system_prompt = self._build_system_prompt(context_text)

            # Build conversation messages
            messages = self._build_conversation_messages(
                system_prompt, chat_history, user_message
            )

            logger.info(f"🤖 Generating AI response with {len(context_chunks)} context chunks")

            # Call OpenAI API
            start_time = time.time()
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )

            # Extract response
            ai_message = response.choices[0].message.content
            response_time = (time.time() - start_time) * 1000

            # Prepare metadata
            metadata = {
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "response_time_ms": round(response_time, 2),
                "context_chunks_used": len(context_chunks),
                "finish_reason": response.choices[0].finish_reason
            }

            logger.info(f"✅ AI response generated: {metadata['tokens_used']} tokens in {response_time:.2f}ms")
            return ai_message, metadata

        except Exception as e:
            logger.error(f"❌ AI response generation failed: {str(e)}")
            raise OpenAIError(f"Failed to generate AI response: {str(e)}")

    def _build_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Build context text from retrieved chunks."""
        if not context_chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk.get("content", "")
            document_info = chunk.get("document_info", {})
            filename = document_info.get("filename", "Document")
            chunk_type = chunk.get("chunk_type", "regular")

            # Format chunk with metadata
            chunk_header = f"[Source {i}: {filename}"
            if chunk_type == "qa_pair":
                chunk_header += " - Q&A"
            chunk_header += "]"

            context_parts.append(f"{chunk_header}\n{content}")

        return "\n\n".join(context_parts)

    def _build_system_prompt(self, context_text: str) -> str:
        """Build system prompt for the AI assistant."""
        base_prompt = f"""Tu es Akissi, l'assistante virtuelle de BMI (Bureau Médical International). Tu es professionnelle, serviable et experte dans tous les services BMI.

INSTRUCTIONS IMPORTANTES:
1. Réponds TOUJOURS en français, même si la question est en anglais
2. Utilise UNIQUEMENT les informations fournies dans le contexte ci-dessous
3. Si l'information n'est pas dans le contexte, dis clairement que tu n'as pas cette information
4. Sois précise, claire et professionnelle
5. Pour les questions sur les assurances, réclamations, ou services BMI, utilise les informations du contexte
6. Si c'est une question générale de politesse, réponds naturellement

CONTEXTE DOCUMENTAIRE:
{context_text if context_text else "Aucun contexte spécifique disponible pour cette question."}

Réponds maintenant à la question de l'utilisateur en utilisant ce contexte."""

        return base_prompt

    def _build_conversation_messages(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str
    ) -> List[Dict[str, str]]:
        """Build conversation messages for OpenAI API."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history (limit to prevent token overflow)
        max_history_tokens = 2000  # Rough estimate
        current_tokens = len(system_prompt) // 4  # Rough token estimation

        for msg in chat_history:
            msg_tokens = len(msg["content"]) // 4
            if current_tokens + msg_tokens > max_history_tokens:
                break
            messages.append(msg)
            current_tokens += msg_tokens

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def _save_ai_message(
        self,
        ai_response: str,
        chat_session: ChatSession,
        metadata: Dict[str, Any],
        context_chunks: List[Dict[str, Any]],
        db_session: AsyncSession
    ) -> ChatMessage:
        """Save AI response to database."""
        # Prepare source documents and chunks
        source_documents = []
        retrieved_chunks = []

        for chunk in context_chunks:
            doc_info = chunk.get("document_info", {})
            if doc_info.get("id"):
                source_documents.append(doc_info["id"])

            retrieved_chunks.append({
                "chunk_id": chunk.get("metadata", {}).get("chunk_index"),
                "document_id": doc_info.get("id"),
                "score": chunk.get("score", 0.0)
            })

        ai_message = ChatMessage(
            id=str(uuid4()),
            session_id=chat_session.id,
            role=MessageRole.ASSISTANT,
            content=ai_response,
            message_index=chat_session.message_count,
            model_name=metadata.get("model"),
            tokens_used=metadata.get("tokens_used"),
            cost=self._calculate_cost(metadata.get("tokens_used", 0)),
            response_time_ms=int(metadata.get("response_time_ms", 0)),
            retrieved_chunks=json.dumps(retrieved_chunks),
            source_documents=json.dumps(source_documents),
            timestamp=datetime.utcnow()
        )

        db_session.add(ai_message)
        chat_session.add_message_stats(
            tokens_used=metadata.get("tokens_used", 0),
            cost=ai_message.cost
        )
        await db_session.commit()

        return ai_message

    async def _save_error_message(
        self,
        error_message: str,
        chat_session: ChatSession,
        db_session: AsyncSession
    ) -> None:
        """Save error message for debugging."""
        error_msg = ChatMessage(
            id=str(uuid4()),
            session_id=chat_session.id,
            role=MessageRole.ASSISTANT,
            content=f"Désolée, j'ai rencontré une erreur: {error_message}",
            message_index=chat_session.message_count,
            timestamp=datetime.utcnow()
        )

        db_session.add(error_msg)
        chat_session.add_message_stats()
        await db_session.commit()

    async def _update_session_stats(
        self,
        chat_session: ChatSession,
        metadata: Dict[str, Any],
        db_session: AsyncSession
    ) -> None:
        """Update session statistics."""
        # Update user activity
        user = await db_session.get(User, chat_session.user_id)
        if user:
            user.increment_message_count()

        await db_session.commit()

    def _calculate_cost(self, total_tokens: int) -> float:
        """Calculate cost based on token usage for GPT-4o."""
        # GPT-4o pricing (as of 2024)
        # Input: $5.00 / 1M tokens
        # Output: $15.00 / 1M tokens
        # Using average cost for simplicity
        cost_per_1k_tokens = 0.01  # $0.01 per 1K tokens (average)
        return (total_tokens / 1000) * cost_per_1k_tokens

    async def get_session_summary(
        self,
        session_id: str,
        db_session: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Get session summary and statistics."""
        try:
            chat_session = await db_session.get(ChatSession, session_id)
            if not chat_session:
                return None

            # Get user info
            user = await db_session.get(User, chat_session.user_id)

            summary = {
                "session_id": session_id,
                "status": chat_session.status.value,
                "started_at": chat_session.started_at.isoformat(),
                "last_message_at": chat_session.last_message_at.isoformat(),
                "message_count": chat_session.message_count,
                "total_tokens_used": chat_session.total_tokens_used,
                "total_cost": chat_session.total_cost,
                "average_tokens_per_message": chat_session.average_tokens_per_message,
                "duration_minutes": chat_session.duration_minutes,
                "user_info": {
                    "preferred_language": user.preferred_language if user else "fr",
                    "total_sessions": user.total_sessions if user else 1,
                    "total_messages": user.total_messages if user else 0
                }
            }

            return summary

        except Exception as e:
            logger.error(f"❌ Failed to get session summary: {str(e)}")
            return None

    async def end_session(
        self,
        session_id: str,
        db_session: AsyncSession
    ) -> bool:
        """End a chat session."""
        try:
            chat_session = await db_session.get(ChatSession, session_id)
            if not chat_session:
                return False

            chat_session.end_session()
            await db_session.commit()

            logger.info(f"✅ Session ended: {session_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to end session: {str(e)}")
            return False
