"""LangChain Financial Retriever.

A production-ready hybrid retriever optimized for financial documents
including SEC filings (10-K, 10-Q, 8-K) and earnings call transcripts.

Features:
- Hybrid search (dense + sparse) with Reciprocal Rank Fusion
- Financial entity extraction and normalization
- Metadata filtering (document type, company, fiscal period)
- Production-ready error handling and logging

Example:
    Basic usage:

    >>> from langchain_financial_retriever import HybridFinancialRetriever
    >>> retriever = HybridFinancialRetriever(
    ...     vectorstore=vectorstore,
    ...     k=10
    ... )
    >>> docs = retriever.invoke("What are Apple's risk factors?")

    Advanced usage with filtering:

    >>> retriever = HybridFinancialRetriever(
    ...     vectorstore=vectorstore,
    ...     k=10,
    ...     document_types=["10-K", "10-Q"],
    ...     fiscal_years=[2024, 2023]
    ... )
    >>> docs = retriever.invoke("revenue growth")
"""

from .hybrid_financial import HybridFinancialRetriever
from .preprocessing import QueryPreprocessor
from .fusion import FusionRanker
from .filtering import MetadataFilter

__version__ = "0.1.0"

__all__ = [
    "HybridFinancialRetriever",
    "QueryPreprocessor",
    "FusionRanker",
    "MetadataFilter",
]
