"""Hybrid financial document retriever for LangChain.

This module provides a production-ready retriever optimized for financial
documents including SEC filings (10-K, 10-Q, 8-K) and earnings call transcripts.
"""

import asyncio
from typing import List, Optional, Any, Dict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from pydantic import Field, ConfigDict, model_validator

from .preprocessing import QueryPreprocessor
from .fusion import FusionRanker
from .filtering import MetadataFilter


class HybridFinancialRetriever(BaseRetriever):
    """Hybrid retriever optimized for financial documents.

    Combines dense retrieval (embeddings) and sparse retrieval (BM25) with
    Reciprocal Rank Fusion for optimal ranking. Specifically designed for
    financial document types including SEC filings and earnings calls.

    Key Features:
    - Hybrid search (semantic + keyword) with RRF fusion
    - Financial entity extraction and normalization
    - Metadata filtering (document type, company, fiscal period)
    - Optional cross-encoder reranking
    - Production-ready error handling and logging

    Optimized for:
    - SEC filings (10-K, 10-Q, 8-K)
    - Earnings call transcripts
    - Financial research reports
    - Credit analysis documents

    Example:
        Basic usage with vector store:

        .. code-block:: python

            from langchain_financial_retriever import HybridFinancialRetriever
            from langchain_community.vectorstores import FAISS

            retriever = HybridFinancialRetriever(
                vectorstore=vectorstore,
                k=10
            )
            docs = retriever.invoke("What are Apple's risk factors?")

        Advanced usage with filtering:

        .. code-block:: python

            retriever = HybridFinancialRetriever(
                vectorstore=vectorstore,
                k=10,
                dense_weight=0.6,
                sparse_weight=0.4,
                document_types=["10-K", "10-Q"],
                fiscal_years=[2024, 2023],
                sections=["Risk Factors", "MD&A"]
            )

        Use with RetrievalQA chain:

        .. code-block:: python

            from langchain.chains import RetrievalQA
            from langchain_openai import ChatOpenAI

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4"),
                retriever=retriever
            )
            result = qa_chain.invoke({"query": "What are the main risks?"})

    Related:
        - BaseRetriever: https://python.langchain.com/api_reference/core/retrievers.html
        - SEC EDGAR: https://www.sec.gov/edgar
        - Financial Document Processing: https://www.sec.gov/how-to-search-edgar
    """

    # Core components
    vectorstore: Any = Field(
        description="Vector store for dense retrieval (semantic search)"
    )
    documents: Optional[List[Document]] = Field(
        default=None,
        description="Document list for sparse retrieval (BM25). "
        "If None, only dense retrieval is used."
    )

    # Retrieval parameters
    k: int = Field(
        default=10,
        ge=1,
        description="Number of documents to return in final results"
    )
    dense_k: int = Field(
        default=20,
        description="Number of documents to retrieve from dense search"
    )
    sparse_k: int = Field(
        default=20,
        description="Number of documents to retrieve from sparse search"
    )

    # Fusion parameters
    dense_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for dense retrieval (0-1). Higher favors semantic search."
    )
    sparse_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for sparse retrieval (0-1). Higher favors keyword matching."
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant for rank fusion (standard value: 60)"
    )

    # Metadata filters
    document_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by document type: ['10-K', '10-Q', '8-K', 'earnings_call']"
    )
    companies: Optional[List[str]] = Field(
        default=None,
        description="Filter by company name or ticker (case-insensitive)"
    )
    fiscal_years: Optional[List[int]] = Field(
        default=None,
        description="Filter by fiscal year (e.g., [2024, 2023])"
    )
    quarters: Optional[List[str]] = Field(
        default=None,
        description="Filter by quarter (e.g., ['Q1', 'Q2'])"
    )
    sections: Optional[List[str]] = Field(
        default=None,
        description="Filter by section: ['Risk Factors', 'MD&A', 'Financial Statements']"
    )

    # Advanced options
    normalize_numbers: bool = Field(
        default=True,
        description="Normalize number formats in query ($5.2M → 5200000)"
    )
    extract_entities: bool = Field(
        default=True,
        description="Extract financial entities from query (tickers, dates, etc.)"
    )
    expand_synonyms: bool = Field(
        default=False,
        description="Expand query with financial synonyms (may increase recall, lower precision)"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging output"
    )

    # Private components (initialized after construction)
    _preprocessor: Optional[QueryPreprocessor] = None
    _fusion_ranker: Optional[FusionRanker] = None
    _metadata_filter: Optional[MetadataFilter] = None
    _bm25_retriever: Optional[Any] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_components(self) -> "HybridFinancialRetriever":
        """Initialize helper components after construction."""
        # Initialize query preprocessor
        self._preprocessor = QueryPreprocessor(
            extract_entities=self.extract_entities,
            normalize_numbers=self.normalize_numbers,
            expand_synonyms=self.expand_synonyms,
        )

        # Initialize fusion ranker
        self._fusion_ranker = FusionRanker(
            weights=[self.dense_weight, self.sparse_weight],
            k=self.rrf_k,
        )

        # Initialize metadata filter if any filters specified
        if any([
            self.document_types,
            self.companies,
            self.fiscal_years,
            self.quarters,
            self.sections,
        ]):
            self._metadata_filter = MetadataFilter(
                document_types=self.document_types,
                companies=self.companies,
                fiscal_years=self.fiscal_years,
                quarters=self.quarters,
                sections=self.sections,
            )

        # Initialize BM25 retriever if documents provided
        if self.documents:
            try:
                from rank_bm25 import BM25Okapi
                tokenized_docs = [
                    self._simple_tokenize(doc.page_content)
                    for doc in self.documents
                ]
                self._bm25_retriever = BM25Okapi(tokenized_docs)
            except ImportError:
                # BM25 not available, will use dense-only
                self._bm25_retriever = None

        return self

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that preserves financial terms.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        import re
        # Preserve tickers, numbers, percentages
        tokens = re.findall(
            r'\$?[\d,]+(?:\.\d+)?[KMB%]?|\b[A-Z]{1,5}\b|\w+',
            text
        )
        return tokens

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents synchronously.

        This is the main retrieval method that combines:
        1. Query preprocessing
        2. Dense retrieval (embeddings)
        3. Sparse retrieval (BM25, if available)
        4. Fusion ranking (RRF)
        5. Metadata filtering
        6. Final selection of top-k

        Args:
            query: Search query string
            run_manager: Callback manager for logging

        Returns:
            List of relevant documents with metadata
        """
        # 1. Query Preprocessing
        run_manager.on_text(f"Preprocessing query: {query}\n", verbose=self.verbose)
        query_info = self._preprocessor.preprocess(query)
        processed_query = query_info["processed_query"]

        if query_info["entities"]:
            run_manager.on_text(
                f"Extracted entities: {query_info['entities']}\n",
                verbose=self.verbose
            )

        # 2. Dense Retrieval (always performed)
        run_manager.on_text("Retrieving with dense search (embeddings)...\n", verbose=self.verbose)
        dense_docs = self._dense_retrieve(processed_query)
        run_manager.on_text(f"Dense retrieval: {len(dense_docs)} docs\n", verbose=self.verbose)

        # 3. Sparse Retrieval (if BM25 available)
        sparse_docs = []
        if self._bm25_retriever and self.documents:
            run_manager.on_text("Retrieving with sparse search (BM25)...\n", verbose=self.verbose)
            sparse_docs = self._sparse_retrieve(processed_query)
            run_manager.on_text(f"Sparse retrieval: {len(sparse_docs)} docs\n", verbose=self.verbose)

        # 4. Fusion Ranking
        if sparse_docs:
            run_manager.on_text("Fusing results with RRF...\n", verbose=self.verbose)
            fused_docs = self._fusion_ranker.fuse([dense_docs, sparse_docs])
        else:
            # Dense only
            fused_docs = dense_docs

        run_manager.on_text(f"After fusion: {len(fused_docs)} docs\n", verbose=self.verbose)

        # 5. Metadata Filtering
        if self._metadata_filter:
            run_manager.on_text("Applying metadata filters...\n", verbose=self.verbose)
            fused_docs = self._metadata_filter.filter(fused_docs)
            run_manager.on_text(f"After filtering: {len(fused_docs)} docs\n", verbose=self.verbose)

        # 6. Take top k
        final_docs = fused_docs[: self.k]

        run_manager.on_text(f"Returning {len(final_docs)} documents\n", verbose=self.verbose)
        return final_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents asynchronously.

        Implements true async parallelism for dense and sparse retrieval.

        Args:
            query: Search query string
            run_manager: Async callback manager for logging

        Returns:
            List of relevant documents with metadata
        """
        # 1. Query Preprocessing (fast, run inline)
        await run_manager.on_text(f"Preprocessing query: {query}\n", verbose=self.verbose)
        query_info = self._preprocessor.preprocess(query)
        processed_query = query_info["processed_query"]

        if query_info["entities"]:
            await run_manager.on_text(
                f"Extracted entities: {query_info['entities']}\n",
                verbose=self.verbose
            )

        # 2. Parallel Retrieval (true async!)
        await run_manager.on_text(
            "Retrieving from dense and sparse retrievers in parallel...\n",
            verbose=self.verbose
        )

        tasks = []

        # Dense retrieval task
        dense_task = asyncio.create_task(self._adense_retrieve(processed_query))
        tasks.append(dense_task)

        # Sparse retrieval task (if available)
        if self._bm25_retriever and self.documents:
            sparse_task = asyncio.create_task(self._asparse_retrieve(processed_query))
            tasks.append(sparse_task)

        # Wait for all retrievals
        results = await asyncio.gather(*tasks)
        dense_docs = results[0]
        sparse_docs = results[1] if len(results) > 1 else []

        await run_manager.on_text(
            f"Dense: {len(dense_docs)}, Sparse: {len(sparse_docs)} docs\n",
            verbose=self.verbose
        )

        # 3. Fusion Ranking (fast, run inline)
        if sparse_docs:
            await run_manager.on_text("Fusing results with RRF...\n", verbose=self.verbose)
            fused_docs = self._fusion_ranker.fuse([dense_docs, sparse_docs])
        else:
            fused_docs = dense_docs

        await run_manager.on_text(f"After fusion: {len(fused_docs)} docs\n", verbose=self.verbose)

        # 4. Metadata Filtering (fast, run inline)
        if self._metadata_filter:
            await run_manager.on_text("Applying metadata filters...\n", verbose=self.verbose)
            fused_docs = self._metadata_filter.filter(fused_docs)
            await run_manager.on_text(
                f"After filtering: {len(fused_docs)} docs\n",
                verbose=self.verbose
            )

        # 5. Take top k
        final_docs = fused_docs[: self.k]

        await run_manager.on_text(
            f"Returning {len(final_docs)} documents\n",
            verbose=self.verbose
        )
        return final_docs

    def _dense_retrieve(self, query: str) -> List[Document]:
        """Perform dense retrieval using vector store.

        Args:
            query: Processed query string

        Returns:
            List of documents from dense retrieval
        """
        docs = self.vectorstore.similarity_search(query, k=self.dense_k)

        # Add dense score to metadata if available
        if hasattr(self.vectorstore, "similarity_search_with_relevance_scores"):
            docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=self.dense_k
            )
            for i, (doc, score) in enumerate(docs_with_scores):
                docs[i].metadata["dense_score"] = score

        return docs

    async def _adense_retrieve(self, query: str) -> List[Document]:
        """Async dense retrieval.

        Args:
            query: Processed query string

        Returns:
            List of documents from dense retrieval
        """
        # Check if vectorstore has async method
        if hasattr(self.vectorstore, "asimilarity_search"):
            return await self.vectorstore.asimilarity_search(query, k=self.dense_k)
        else:
            # Fallback to sync in executor
            return await asyncio.to_thread(self._dense_retrieve, query)

    def _sparse_retrieve(self, query: str) -> List[Document]:
        """Perform sparse retrieval using BM25.

        Args:
            query: Processed query string

        Returns:
            List of documents from sparse retrieval
        """
        if not self._bm25_retriever or not self.documents:
            return []

        # Tokenize query
        tokenized_query = self._simple_tokenize(query)

        # Get BM25 scores
        scores = self._bm25_retriever.get_scores(tokenized_query)

        # Get top k indices
        import numpy as np
        top_indices = np.argsort(scores)[-self.sparse_k:][::-1]

        # Build results with scores
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                # Create a copy to avoid modifying original
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata)
                )
                doc_copy.metadata["sparse_score"] = float(scores[idx])
                results.append(doc_copy)

        return results

    async def _asparse_retrieve(self, query: str) -> List[Document]:
        """Async sparse retrieval (BM25 is CPU-bound, run in thread).

        Args:
            query: Processed query string

        Returns:
            List of documents from sparse retrieval
        """
        return await asyncio.to_thread(self._sparse_retrieve, query)
