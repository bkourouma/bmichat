"""
BMI Chat Application - Embedding Service

This module handles OpenAI embedding generation for text chunks.
Provides efficient batch processing and error handling for embeddings.
"""

import asyncio
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings
from app.core.exceptions import OpenAIError


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI's embedding models.
    
    Handles batch processing, rate limiting, and error recovery.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "text-embedding-ada-002"  # OpenAI's latest embedding model
        self.max_batch_size = 100  # OpenAI's batch limit
        self.max_tokens_per_request = 8000  # Approximate token limit
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            OpenAIError: If embedding generation fails
        """
        if not texts:
            return []
        
        try:
            logger.info(f"🧠 Generating embeddings for {len(texts)} texts")
            
            # Process in batches if necessary
            if len(texts) > self.max_batch_size:
                return await self._process_in_batches(texts)
            
            # Single batch processing
            embeddings = await self._generate_batch_embeddings(texts)
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {str(e)}")
            raise OpenAIError(f"Failed to generate embeddings: {str(e)}")
    
    async def _process_in_batches(self, texts: List[str]) -> List[List[float]]:
        """Process texts in batches to respect API limits."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            logger.debug(f"Processing batch {i // self.max_batch_size + 1}")
            
            batch_embeddings = await self._generate_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to respect rate limits
            if i + self.max_batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a single batch of texts."""
        try:
            # Clean and prepare texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Call OpenAI API
            response = await self.client.embeddings.create(
                model=self.model,
                input=cleaned_texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {str(e)}")
            raise OpenAIError(f"Batch embedding generation failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text before embedding generation.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Truncate if too long (approximate token limit)
        max_chars = self.max_tokens_per_request * 4  # Rough estimate: 1 token ≈ 4 chars
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        return cleaned
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from the current model.
        
        Returns:
            Embedding dimension
        """
        try:
            # Generate a test embedding to get dimension
            test_embedding = await self.generate_single_embedding("test")
            return len(test_embedding)
            
        except Exception as e:
            logger.error(f"❌ Failed to get embedding dimension: {str(e)}")
            # Return known dimension for text-embedding-ada-002
            return 1536
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def can_process_batch(self, texts: List[str]) -> bool:
        """
        Check if a batch of texts can be processed within limits.
        
        Args:
            texts: List of texts to check
            
        Returns:
            True if batch can be processed
        """
        if len(texts) > self.max_batch_size:
            return False
        
        total_tokens = sum(self.estimate_tokens(text) for text in texts)
        return total_tokens <= self.max_tokens_per_request
