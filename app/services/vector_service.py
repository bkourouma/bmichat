"""
BMI Chat Application - Vector Service

This module provides vector database operations using ChromaDB.
Handles document embeddings, similarity search, and vector storage.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from loguru import logger

from app.config import settings
from app.core.exceptions import VectorDatabaseError


class VectorService:
    """
    Vector database service using ChromaDB.
    
    Handles document embeddings, storage, and similarity search operations.
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "bmi_documents"
    
    async def initialize(self) -> None:
        """
        Initialize the vector database connection.
        
        Raises:
            VectorDatabaseError: If initialization fails
        """
        try:
            logger.info("🔧 Initializing vector database...")
            
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Create vector database directory
            vector_db_path = Path(settings.vector_db_path)
            vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(vector_db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"✅ Using existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "BMI Chat document embeddings"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
            
            logger.info("✅ Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {str(e)}")
            raise VectorDatabaseError(f"Vector database initialization failed: {str(e)}")
    
    async def add_document_chunks(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add document chunks with embeddings to the vector database.
        
        Args:
            document_id: Unique document identifier
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: List of metadata for each chunk
            
        Raises:
            VectorDatabaseError: If adding chunks fails
        """
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"📄 Adding {len(chunks)} chunks for document {document_id}")
            
            # Generate unique IDs for each chunk
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Add chunks to collection
            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata
            )
            
            logger.info(f"✅ Added {len(chunks)} chunks to vector database")
            
        except Exception as e:
            logger.error(f"❌ Failed to add document chunks: {str(e)}")
            raise VectorDatabaseError(f"Failed to add document chunks: {str(e)}")
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        keywords_filter: Optional[List[str]] = None,
        chunk_type_filter: Optional[str] = None,
        boost_qa_pairs: bool = True
    ) -> Dict[str, Any]:
        """
        Advanced similarity search with metadata filtering and ranking.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filter
            keywords_filter: Filter by document keywords
            chunk_type_filter: Filter by chunk type (qa_pair, regular, etc.)
            boost_qa_pairs: Boost Q&A pairs in ranking

        Returns:
            Dict containing search results with enhanced ranking

        Raises:
            VectorDatabaseError: If search fails
        """
        try:
            if not self.collection:
                await self.initialize()

            k = k or settings.default_retrieval_k
            k = min(k, settings.max_retrieval_k)

            logger.info(f"🔍 Advanced search for {k} similar chunks")

            # Build metadata filter
            where_filter = self._build_metadata_filter(
                filter_metadata, keywords_filter, chunk_type_filter
            )

            # Get more results initially for re-ranking
            search_k = min(k * 3, 50)  # Get 3x results for re-ranking

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_k,
                where=where_filter
            )

            # Enhanced post-processing and ranking
            enhanced_results = await self._enhance_and_rank_results(
                results, query_embedding, k, boost_qa_pairs
            )

            logger.info(f"✅ Found {len(enhanced_results['chunks'])} enhanced results")
            return enhanced_results

        except Exception as e:
            logger.error(f"❌ Failed to search similar chunks: {str(e)}")
            raise VectorDatabaseError(f"Failed to search similar chunks: {str(e)}")

    def _build_metadata_filter(
        self,
        base_filter: Optional[Dict[str, Any]],
        keywords_filter: Optional[List[str]],
        chunk_type_filter: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Build comprehensive metadata filter."""
        where_filter = base_filter.copy() if base_filter else {}

        # Add keywords filter
        if keywords_filter:
            # ChromaDB supports $contains operator for partial matches
            keyword_conditions = []
            for keyword in keywords_filter:
                keyword_conditions.append({"keywords": {"$contains": keyword}})

            if len(keyword_conditions) == 1:
                where_filter.update(keyword_conditions[0])
            else:
                where_filter["$or"] = keyword_conditions

        # Add chunk type filter
        if chunk_type_filter:
            where_filter["chunk_type"] = chunk_type_filter

        return where_filter if where_filter else None

    async def _enhance_and_rank_results(
        self,
        raw_results: Dict[str, Any],
        query_embedding: List[float],
        final_k: int,
        boost_qa_pairs: bool
    ) -> Dict[str, Any]:
        """Enhanced ranking with multiple factors."""
        if not raw_results["documents"] or not raw_results["documents"][0]:
            return {
                "chunks": [],
                "distances": [],
                "metadata": [],
                "ids": [],
                "scores": [],
                "ranking_info": []
            }

        chunks = raw_results["documents"][0]
        distances = raw_results["distances"][0]
        metadatas = raw_results["metadatas"][0]
        ids = raw_results["ids"][0]

        # Calculate enhanced scores
        enhanced_results = []
        for i, (chunk, distance, metadata, chunk_id) in enumerate(zip(chunks, distances, metadatas, ids)):
            # Base similarity score (convert distance to similarity)
            similarity_score = 1.0 - distance

            # Calculate ranking factors
            ranking_factors = self._calculate_ranking_factors(
                chunk, metadata, similarity_score, boost_qa_pairs
            )

            # Combined score
            final_score = self._calculate_final_score(similarity_score, ranking_factors)

            enhanced_results.append({
                "chunk": chunk,
                "distance": distance,
                "metadata": metadata,
                "id": chunk_id,
                "similarity_score": similarity_score,
                "final_score": final_score,
                "ranking_factors": ranking_factors
            })

        # Sort by final score (descending)
        enhanced_results.sort(key=lambda x: x["final_score"], reverse=True)

        # Take top k results
        top_results = enhanced_results[:final_k]

        # Format for return
        return {
            "chunks": [r["chunk"] for r in top_results],
            "distances": [r["distance"] for r in top_results],
            "metadata": [r["metadata"] for r in top_results],
            "ids": [r["id"] for r in top_results],
            "scores": [r["final_score"] for r in top_results],
            "ranking_info": [r["ranking_factors"] for r in top_results]
        }

    def _calculate_ranking_factors(
        self,
        chunk: str,
        metadata: Dict[str, Any],
        similarity_score: float,
        boost_qa_pairs: bool
    ) -> Dict[str, float]:
        """Calculate various ranking factors."""
        factors = {}

        # Chunk type boost
        chunk_type = metadata.get("chunk_type", "regular")
        if boost_qa_pairs and chunk_type == "qa_pair":
            factors["qa_boost"] = 0.15  # 15% boost for Q&A pairs
        else:
            factors["qa_boost"] = 0.0

        # Content quality factors
        has_questions = metadata.get("has_questions", False)
        has_answers = metadata.get("has_answers", False)

        if has_questions and has_answers:
            factors["qa_completeness"] = 0.1  # 10% boost for complete Q&A
        elif has_questions or has_answers:
            factors["qa_completeness"] = 0.05  # 5% boost for partial Q&A
        else:
            factors["qa_completeness"] = 0.0

        # Length factor (prefer moderate length chunks)
        chunk_length = len(chunk)
        if 500 <= chunk_length <= 2000:
            factors["length_factor"] = 0.05  # 5% boost for optimal length
        elif chunk_length < 100:
            factors["length_factor"] = -0.1  # Penalty for very short chunks
        else:
            factors["length_factor"] = 0.0

        # Confidence score from chunking
        confidence = metadata.get("confidence_score", 1.0)
        factors["confidence_factor"] = (confidence - 1.0) * 0.1  # Up to 10% boost

        # Recency factor (newer documents get slight boost)
        # This could be enhanced with actual document dates
        factors["recency_factor"] = 0.0

        return factors

    def _calculate_final_score(
        self,
        similarity_score: float,
        ranking_factors: Dict[str, float]
    ) -> float:
        """Calculate final ranking score."""
        # Start with similarity score
        final_score = similarity_score

        # Apply all ranking factors
        for factor_name, factor_value in ranking_factors.items():
            final_score += factor_value

        # Ensure score stays in reasonable bounds
        return max(0.0, min(1.0, final_score))
    
    async def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete
            
        Raises:
            VectorDatabaseError: If deletion fails
        """
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"🗑️ Deleting document chunks: {document_id}")
            
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results["ids"]:
                # Delete the chunks
                self.collection.delete(ids=results["ids"])
                logger.info(f"✅ Deleted {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.info(f"ℹ️ No chunks found for document {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {str(e)}")
            raise VectorDatabaseError(f"Failed to delete document: {str(e)}")
    
    async def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents in the collection
            
        Raises:
            VectorDatabaseError: If count retrieval fails
        """
        try:
            if not self.collection:
                await self.initialize()
            
            count = self.collection.count()
            logger.debug(f"📊 Collection contains {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection count: {str(e)}")
            raise VectorDatabaseError(f"Failed to get collection count: {str(e)}")
    
    async def reset_collection(self) -> None:
        """
        Reset the entire collection (delete all documents).
        
        Warning: This will delete all stored embeddings!
        
        Raises:
            VectorDatabaseError: If reset fails
        """
        try:
            if not self.client:
                await self.initialize()
            
            logger.warning("⚠️ Resetting vector database collection...")
            
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "BMI Chat document embeddings"}
            )
            
            logger.info("✅ Vector database collection reset successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to reset collection: {str(e)}")
            raise VectorDatabaseError(f"Failed to reset collection: {str(e)}")

    async def search_by_keywords(
        self,
        keywords: List[str],
        k: int = None,
        chunk_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search documents by keywords without embedding similarity.

        Args:
            keywords: List of keywords to search for
            k: Number of results to return
            chunk_type_filter: Filter by chunk type

        Returns:
            Dict containing keyword-based search results
        """
        try:
            if not self.collection:
                await self.initialize()

            k = k or settings.default_retrieval_k

            logger.info(f"🔍 Keyword search for: {keywords}")

            # Build keyword filter
            where_filter = self._build_metadata_filter(
                None, keywords, chunk_type_filter
            )

            # Get all matching documents (no embedding query)
            results = self.collection.get(
                where=where_filter,
                limit=k
            )

            # Format results
            formatted_results = {
                "chunks": results["documents"] if results["documents"] else [],
                "metadata": results["metadatas"] if results["metadatas"] else [],
                "ids": results["ids"] if results["ids"] else [],
                "distances": [0.0] * len(results["documents"]) if results["documents"] else [],
                "scores": [1.0] * len(results["documents"]) if results["documents"] else []
            }

            logger.info(f"✅ Found {len(formatted_results['chunks'])} keyword matches")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Failed to search by keywords: {str(e)}")
            raise VectorDatabaseError(f"Failed to search by keywords: {str(e)}")

    async def get_document_chunks(
        self,
        document_id: str,
        chunk_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: Document ID
            chunk_type_filter: Optional chunk type filter

        Returns:
            Dict containing document chunks
        """
        try:
            if not self.collection:
                await self.initialize()

            logger.info(f"📄 Getting chunks for document: {document_id}")

            # Build filter
            where_filter = {"document_id": document_id}
            if chunk_type_filter:
                where_filter["chunk_type"] = chunk_type_filter

            # Get document chunks
            results = self.collection.get(
                where=where_filter
            )

            # Sort by chunk_index if available
            if results["metadatas"]:
                # Combine results for sorting
                combined = list(zip(
                    results["documents"],
                    results["metadatas"],
                    results["ids"]
                ))

                # Sort by chunk_index
                combined.sort(key=lambda x: x[1].get("chunk_index", 0))

                # Unpack sorted results
                sorted_docs, sorted_metadata, sorted_ids = zip(*combined) if combined else ([], [], [])

                formatted_results = {
                    "chunks": list(sorted_docs),
                    "metadata": list(sorted_metadata),
                    "ids": list(sorted_ids),
                    "distances": [0.0] * len(sorted_docs),
                    "scores": [1.0] * len(sorted_docs)
                }
            else:
                formatted_results = {
                    "chunks": [],
                    "metadata": [],
                    "ids": [],
                    "distances": [],
                    "scores": []
                }

            logger.info(f"✅ Retrieved {len(formatted_results['chunks'])} chunks")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Failed to get document chunks: {str(e)}")
            raise VectorDatabaseError(f"Failed to get document chunks: {str(e)}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive collection statistics.

        Returns:
            Dict with collection statistics
        """
        try:
            if not self.collection:
                await self.initialize()

            logger.info("📊 Getting collection statistics")

            # Get all documents for analysis
            all_results = self.collection.get()

            if not all_results["metadatas"]:
                return {
                    "total_chunks": 0,
                    "total_documents": 0,
                    "chunk_types": {},
                    "keywords": {},
                    "average_chunk_length": 0,
                    "qa_coverage": 0
                }

            metadatas = all_results["metadatas"]
            documents = all_results["documents"]

            # Calculate statistics
            total_chunks = len(metadatas)
            unique_documents = len(set(meta.get("document_id", "") for meta in metadatas))

            # Chunk type distribution
            chunk_types = {}
            for meta in metadatas:
                chunk_type = meta.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            # Keywords distribution
            keywords_count = {}
            for meta in metadatas:
                keywords = meta.get("keywords", "")
                if keywords:
                    for keyword in keywords.split(","):
                        keyword = keyword.strip()
                        if keyword:
                            keywords_count[keyword] = keywords_count.get(keyword, 0) + 1

            # Average chunk length
            total_length = sum(len(doc) for doc in documents)
            avg_length = total_length / total_chunks if total_chunks > 0 else 0

            # Q&A coverage
            qa_chunks = chunk_types.get("qa_pair", 0)
            qa_coverage = (qa_chunks / total_chunks * 100) if total_chunks > 0 else 0

            stats = {
                "total_chunks": total_chunks,
                "total_documents": unique_documents,
                "chunk_types": chunk_types,
                "top_keywords": dict(sorted(keywords_count.items(), key=lambda x: x[1], reverse=True)[:10]),
                "average_chunk_length": round(avg_length, 2),
                "qa_coverage": round(qa_coverage, 2),
                "has_questions_count": sum(1 for meta in metadatas if meta.get("has_questions", False)),
                "has_answers_count": sum(1 for meta in metadatas if meta.get("has_answers", False))
            }

            logger.info(f"✅ Collection stats: {total_chunks} chunks, {unique_documents} documents")
            return stats

        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {str(e)}")
            raise VectorDatabaseError(f"Failed to get collection stats: {str(e)}")

    async def hybrid_search(
        self,
        query_embedding: List[float],
        keywords: Optional[List[str]] = None,
        k: int = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        Hybrid search combining semantic similarity and keyword matching.

        Args:
            query_embedding: Query embedding vector
            keywords: Optional keywords for hybrid search
            k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)

        Returns:
            Dict containing hybrid search results
        """
        try:
            if not self.collection:
                await self.initialize()

            k = k or settings.default_retrieval_k

            logger.info(f"🔍 Hybrid search with semantic_weight={semantic_weight}, keyword_weight={keyword_weight}")

            # Normalize weights
            total_weight = semantic_weight + keyword_weight
            if total_weight > 0:
                semantic_weight = semantic_weight / total_weight
                keyword_weight = keyword_weight / total_weight

            # Get semantic results
            semantic_results = await self.search_similar_chunks(
                query_embedding=query_embedding,
                k=k * 2,  # Get more results for merging
                boost_qa_pairs=True
            )

            # Get keyword results if keywords provided
            keyword_results = None
            if keywords:
                keyword_results = await self.search_by_keywords(
                    keywords=keywords,
                    k=k * 2
                )

            # Combine and re-rank results
            hybrid_results = self._combine_hybrid_results(
                semantic_results, keyword_results, semantic_weight, keyword_weight, k
            )

            logger.info(f"✅ Hybrid search returned {len(hybrid_results['chunks'])} results")
            return hybrid_results

        except Exception as e:
            logger.error(f"❌ Failed to perform hybrid search: {str(e)}")
            raise VectorDatabaseError(f"Failed to perform hybrid search: {str(e)}")

    def _combine_hybrid_results(
        self,
        semantic_results: Dict[str, Any],
        keyword_results: Optional[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        final_k: int
    ) -> Dict[str, Any]:
        """Combine and re-rank semantic and keyword results."""
        # Start with semantic results
        combined_scores = {}

        # Add semantic scores
        for i, chunk_id in enumerate(semantic_results["ids"]):
            semantic_score = semantic_results["scores"][i] if i < len(semantic_results["scores"]) else 0.5
            combined_scores[chunk_id] = {
                "semantic_score": semantic_score,
                "keyword_score": 0.0,
                "chunk": semantic_results["chunks"][i],
                "metadata": semantic_results["metadata"][i],
                "distance": semantic_results["distances"][i]
            }

        # Add keyword scores
        if keyword_results:
            for i, chunk_id in enumerate(keyword_results["ids"]):
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]["keyword_score"] = 1.0
                else:
                    combined_scores[chunk_id] = {
                        "semantic_score": 0.0,
                        "keyword_score": 1.0,
                        "chunk": keyword_results["chunks"][i],
                        "metadata": keyword_results["metadata"][i],
                        "distance": 1.0  # Max distance for keyword-only results
                    }

        # Calculate final scores
        for chunk_id, scores in combined_scores.items():
            final_score = (
                scores["semantic_score"] * semantic_weight +
                scores["keyword_score"] * keyword_weight
            )
            scores["final_score"] = final_score

        # Sort by final score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True
        )

        # Take top k results
        top_results = sorted_results[:final_k]

        # Format for return
        return {
            "chunks": [result[1]["chunk"] for result in top_results],
            "distances": [result[1]["distance"] for result in top_results],
            "metadata": [result[1]["metadata"] for result in top_results],
            "ids": [result[0] for result in top_results],
            "scores": [result[1]["final_score"] for result in top_results],
            "semantic_scores": [result[1]["semantic_score"] for result in top_results],
            "keyword_scores": [result[1]["keyword_score"] for result in top_results]
        }
