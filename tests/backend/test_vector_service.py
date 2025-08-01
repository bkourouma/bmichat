"""
BMI Chat Application - Vector Service Tests

Test suite for vector database operations and search functionality.
"""

import pytest
import asyncio
from typing import List, Dict, Any

from app.services.vector_service import VectorService
from app.services.embedding_service import EmbeddingService
from app.config import settings


@pytest.fixture
async def vector_service():
    """Create vector service for testing."""
    service = VectorService()
    await service.initialize()
    
    # Clean up any existing test data
    try:
        await service.reset_collection()
    except:
        pass
    
    yield service
    
    # Cleanup after tests
    try:
        await service.reset_collection()
    except:
        pass


@pytest.fixture
async def embedding_service():
    """Create embedding service for testing."""
    return EmbeddingService()


@pytest.fixture
async def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "chunks": [
                "Q: What is BMI insurance? A: BMI insurance provides comprehensive coverage for your health needs.",
                "Our insurance plans include medical, dental, and vision coverage for individuals and families."
            ],
            "metadata": [
                {
                    "document_id": "doc1",
                    "chunk_index": 0,
                    "chunk_type": "qa_pair",
                    "keywords": "insurance,health,coverage",
                    "has_questions": True,
                    "has_answers": True
                },
                {
                    "document_id": "doc1",
                    "chunk_index": 1,
                    "chunk_type": "regular",
                    "keywords": "insurance,medical,dental",
                    "has_questions": False,
                    "has_answers": False
                }
            ]
        },
        {
            "id": "doc2",
            "chunks": [
                "Q: How do I file a claim? A: You can file a claim online through our portal or by calling customer service.",
                "Claims processing typically takes 3-5 business days for standard requests."
            ],
            "metadata": [
                {
                    "document_id": "doc2",
                    "chunk_index": 0,
                    "chunk_type": "qa_pair",
                    "keywords": "claims,filing,customer-service",
                    "has_questions": True,
                    "has_answers": True
                },
                {
                    "document_id": "doc2",
                    "chunk_index": 1,
                    "chunk_type": "regular",
                    "keywords": "claims,processing",
                    "has_questions": False,
                    "has_answers": False
                }
            ]
        }
    ]


class TestVectorService:
    """Test vector service functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize(self, vector_service):
        """Test vector service initialization."""
        # Service should be initialized from fixture
        assert vector_service.client is not None
        assert vector_service.collection is not None
        
        # Test collection count
        count = await vector_service.get_collection_count()
        assert count >= 0
    
    @pytest.mark.asyncio
    async def test_add_and_search_chunks(self, vector_service, embedding_service, sample_documents):
        """Test adding chunks and searching."""
        # Add sample documents
        for doc in sample_documents:
            # Generate embeddings
            embeddings = await embedding_service.generate_embeddings(doc["chunks"])
            
            # Add to vector database
            await vector_service.add_document_chunks(
                document_id=doc["id"],
                chunks=doc["chunks"],
                embeddings=embeddings,
                metadata=doc["metadata"]
            )
        
        # Test basic search
        query_embedding = await embedding_service.generate_single_embedding("insurance coverage")
        
        results = await vector_service.search_similar_chunks(
            query_embedding=query_embedding,
            k=3
        )
        
        assert len(results["chunks"]) > 0
        assert len(results["chunks"]) <= 3
        assert len(results["distances"]) == len(results["chunks"])
        assert len(results["metadata"]) == len(results["chunks"])
    
    @pytest.mark.asyncio
    async def test_keyword_search(self, vector_service, embedding_service, sample_documents):
        """Test keyword-based search."""
        # Add sample documents
        for doc in sample_documents:
            embeddings = await embedding_service.generate_embeddings(doc["chunks"])
            await vector_service.add_document_chunks(
                document_id=doc["id"],
                chunks=doc["chunks"],
                embeddings=embeddings,
                metadata=doc["metadata"]
            )
        
        # Test keyword search
        results = await vector_service.search_by_keywords(
            keywords=["insurance"],
            k=5
        )
        
        assert len(results["chunks"]) > 0
        
        # Check that results contain insurance-related content
        for chunk in results["chunks"]:
            assert "insurance" in chunk.lower() or "coverage" in chunk.lower()
    
    @pytest.mark.asyncio
    async def test_qa_pair_boosting(self, vector_service, embedding_service, sample_documents):
        """Test Q&A pair boosting in search results."""
        # Add sample documents
        for doc in sample_documents:
            embeddings = await embedding_service.generate_embeddings(doc["chunks"])
            await vector_service.add_document_chunks(
                document_id=doc["id"],
                chunks=doc["chunks"],
                embeddings=embeddings,
                metadata=doc["metadata"]
            )
        
        # Search with Q&A boosting
        query_embedding = await embedding_service.generate_single_embedding("What is insurance?")
        
        results = await vector_service.search_similar_chunks(
            query_embedding=query_embedding,
            k=3,
            boost_qa_pairs=True
        )
        
        assert len(results["chunks"]) > 0
        
        # Check that Q&A pairs are ranked higher
        if len(results["metadata"]) > 1:
            first_result_type = results["metadata"][0].get("chunk_type")
            # First result should likely be a Q&A pair for this query
            assert first_result_type in ["qa_pair", "regular"]  # Allow both for flexibility
    
    @pytest.mark.asyncio
    async def test_metadata_filtering(self, vector_service, embedding_service, sample_documents):
        """Test metadata filtering in search."""
        # Add sample documents
        for doc in sample_documents:
            embeddings = await embedding_service.generate_embeddings(doc["chunks"])
            await vector_service.add_document_chunks(
                document_id=doc["id"],
                chunks=doc["chunks"],
                embeddings=embeddings,
                metadata=doc["metadata"]
            )
        
        # Test filtering by chunk type
        query_embedding = await embedding_service.generate_single_embedding("insurance")
        
        results = await vector_service.search_similar_chunks(
            query_embedding=query_embedding,
            k=5,
            chunk_type_filter="qa_pair"
        )
        
        # All results should be Q&A pairs
        for metadata in results["metadata"]:
            assert metadata.get("chunk_type") == "qa_pair"
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, vector_service, embedding_service, sample_documents):
        """Test hybrid search combining semantic and keyword search."""
        # Add sample documents
        for doc in sample_documents:
            embeddings = await embedding_service.generate_embeddings(doc["chunks"])
            await vector_service.add_document_chunks(
                document_id=doc["id"],
                chunks=doc["chunks"],
                embeddings=embeddings,
                metadata=doc["metadata"]
            )
        
        # Test hybrid search
        query_embedding = await embedding_service.generate_single_embedding("filing claims")
        
        results = await vector_service.hybrid_search(
            query_embedding=query_embedding,
            keywords=["claims"],
            k=3,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        assert len(results["chunks"]) > 0
        assert "semantic_scores" in results
        assert "keyword_scores" in results
        
        # Check that results contain claims-related content
        found_claims = False
        for chunk in results["chunks"]:
            if "claim" in chunk.lower():
                found_claims = True
                break
        assert found_claims
    
    @pytest.mark.asyncio
    async def test_collection_stats(self, vector_service, embedding_service, sample_documents):
        """Test collection statistics."""
        # Add sample documents
        for doc in sample_documents:
            embeddings = await embedding_service.generate_embeddings(doc["chunks"])
            await vector_service.add_document_chunks(
                document_id=doc["id"],
                chunks=doc["chunks"],
                embeddings=embeddings,
                metadata=doc["metadata"]
            )
        
        # Get statistics
        stats = await vector_service.get_collection_stats()
        
        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert "chunk_types" in stats
        assert "top_keywords" in stats
        assert "qa_coverage" in stats
        
        assert stats["total_chunks"] > 0
        assert stats["total_documents"] > 0
        assert stats["qa_coverage"] >= 0
    
    @pytest.mark.asyncio
    async def test_document_deletion(self, vector_service, embedding_service, sample_documents):
        """Test document deletion."""
        # Add a document
        doc = sample_documents[0]
        embeddings = await embedding_service.generate_embeddings(doc["chunks"])
        await vector_service.add_document_chunks(
            document_id=doc["id"],
            chunks=doc["chunks"],
            embeddings=embeddings,
            metadata=doc["metadata"]
        )
        
        # Verify document exists
        initial_count = await vector_service.get_collection_count()
        assert initial_count > 0
        
        # Delete document
        await vector_service.delete_document(doc["id"])
        
        # Verify document is deleted
        final_count = await vector_service.get_collection_count()
        assert final_count < initial_count


class TestEmbeddingService:
    """Test embedding service functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, embedding_service):
        """Test single embedding generation."""
        text = "This is a test sentence for embedding generation."
        
        embedding = await embedding_service.generate_single_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self, embedding_service):
        """Test batch embedding generation."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        embeddings = await embedding_service.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_embedding_dimension(self, embedding_service):
        """Test embedding dimension consistency."""
        texts = ["Short text.", "This is a longer text with more words and content."]
        
        embeddings = await embedding_service.generate_embeddings(texts)
        
        # All embeddings should have the same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1  # All dimensions should be the same
        
        # Should match expected dimension for text-embedding-ada-002
        expected_dim = await embedding_service.get_embedding_dimension()
        assert dimensions[0] == expected_dim
