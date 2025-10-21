"""Reciprocal Rank Fusion for combining retrieval results.

This module implements the RRF algorithm for fusing results from
multiple retrievers (dense + sparse).
"""

from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from langchain_core.documents import Document


class FusionRanker:
    """Combine multiple retrieval results using Reciprocal Rank Fusion (RRF).

    RRF Formula: score = weight / (k + rank + 1)

    Where:
    - rank: Position in the retrieval list (0-indexed)
    - k: Constant (default 60) to balance rank importance
    - weight: Importance of this retriever

    Benefits:
    - No score normalization needed
    - Works with any ranking function
    - Robust to different score scales
    - Handles different result set sizes

    Example:
        >>> ranker = FusionRanker(weights=[0.6, 0.4], k=60)
        >>> dense_docs = [doc1, doc2, doc3]
        >>> sparse_docs = [doc2, doc1, doc4]
        >>> fused = ranker.fuse([dense_docs, sparse_docs])
        >>> # doc2 and doc1 rank highest (appear in both)

    References:
        Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
        Reciprocal rank fusion outperforms condorcet and individual rank
        learning methods. SIGIR '09.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        k: int = 60,
        deduplication_key: str = "source",
    ):
        """Initialize the fusion ranker.

        Args:
            weights: List of weights for each retriever (default: [0.6, 0.4])
                First weight is typically for dense retrieval (semantic)
                Second weight is typically for sparse retrieval (keyword)
            k: RRF constant (default 60, standard value from research)
            deduplication_key: Metadata key to use for deduplication
                Documents with same source+chunk_id are considered duplicates
        """
        self.weights = weights or [0.6, 0.4]  # Default: favor semantic search
        self.k = k
        self.deduplication_key = deduplication_key

    def fuse(self, results_list: List[List[Document]]) -> List[Document]:
        """Fuse multiple ranked lists using RRF.

        Args:
            results_list: List of retrieval results, one list per retriever.
                Each inner list should be sorted by relevance (best first).

        Returns:
            Combined and deduplicated results, sorted by RRF score (descending).
            Each document has 'rrf_score' added to metadata.

        Example:
            >>> dense_results = [doc_a, doc_b, doc_c]
            >>> sparse_results = [doc_c, doc_a, doc_d]
            >>> fused = ranker.fuse([dense_results, sparse_results])
            >>> # Result: [doc_a, doc_c, doc_b, doc_d] with RRF scores
        """
        if not results_list:
            return []

        # Store documents and compute RRF scores
        doc_scores = defaultdict(float)
        doc_map = {}  # Maps doc_key -> Document

        for retriever_idx, docs in enumerate(results_list):
            # Get weight for this retriever
            weight = (
                self.weights[retriever_idx]
                if retriever_idx < len(self.weights)
                else 1.0
            )

            for rank, doc in enumerate(docs):
                # Create unique key for deduplication
                doc_key = self._get_doc_key(doc)

                # Compute RRF score for this retriever
                # Formula: weight / (k + rank + 1)
                # rank is 0-indexed, so first doc is rank 0
                rrf_contribution = weight / (self.k + rank + 1)
                doc_scores[doc_key] += rrf_contribution

                # Store document (keep first occurrence to preserve metadata)
                if doc_key not in doc_map:
                    doc_map[doc_key] = doc

        # Sort by RRF score (descending)
        sorted_keys = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build final result list with RRF scores
        results = []
        for doc_key, rrf_score in sorted_keys:
            doc = doc_map[doc_key]
            # Add RRF score to metadata
            doc.metadata["rrf_score"] = float(rrf_score)
            results.append(doc)

        return results

    def _get_doc_key(self, doc: Document) -> str:
        """Generate unique key for document deduplication.

        Uses source + chunk_id to identify unique chunks.
        If chunk_id is not present, uses hash of page_content.

        Args:
            doc: Document to generate key for

        Returns:
            Unique string key for this document
        """
        source = doc.metadata.get(self.deduplication_key, "")
        chunk_id = doc.metadata.get("chunk_id", "")

        if chunk_id:
            return f"{source}:{chunk_id}"
        else:
            # Fallback: use hash of content for deduplication
            # This handles cases where chunk_id is not set
            content_hash = hash(doc.page_content[:200])  # Use first 200 chars
            return f"{source}:{content_hash}"

    def get_score_breakdown(
        self, doc: Document, results_list: List[List[Document]]
    ) -> Dict[str, Any]:
        """Get detailed RRF score breakdown for a document.

        Useful for debugging and understanding why a document ranked highly.

        Args:
            doc: Document to analyze
            results_list: Original retrieval results

        Returns:
            Dictionary with score breakdown:
                - total_rrf_score: Final RRF score
                - retriever_scores: Score contribution from each retriever
                - retriever_ranks: Rank in each retriever (None if not found)
                - appears_in: List of retriever indices where doc appears

        Example:
            >>> breakdown = ranker.get_score_breakdown(doc, [dense, sparse])
            >>> print(breakdown)
            {
                'total_rrf_score': 0.025,
                'retriever_scores': [0.015, 0.010],
                'retriever_ranks': [0, 2],
                'appears_in': [0, 1]
            }
        """
        doc_key = self._get_doc_key(doc)
        breakdown = {
            "total_rrf_score": 0.0,
            "retriever_scores": [],
            "retriever_ranks": [],
            "appears_in": [],
        }

        for retriever_idx, docs in enumerate(results_list):
            weight = (
                self.weights[retriever_idx]
                if retriever_idx < len(self.weights)
                else 1.0
            )

            # Find document in this retriever's results
            rank = None
            for r, d in enumerate(docs):
                if self._get_doc_key(d) == doc_key:
                    rank = r
                    break

            if rank is not None:
                # Document found in this retriever
                score = weight / (self.k + rank + 1)
                breakdown["retriever_scores"].append(score)
                breakdown["retriever_ranks"].append(rank)
                breakdown["appears_in"].append(retriever_idx)
                breakdown["total_rrf_score"] += score
            else:
                # Document not found in this retriever
                breakdown["retriever_scores"].append(0.0)
                breakdown["retriever_ranks"].append(None)

        return breakdown
