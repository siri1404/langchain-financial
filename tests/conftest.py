"""Pytest configuration and fixtures for langchain_financial tests."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


@pytest.fixture
def mock_vectorstore():
    """Mock vectorstore for testing."""
    mock_vs = Mock(spec=FAISS)
    mock_vs.similarity_search_with_score.return_value = [
        (Mock(page_content="Test document 1", metadata={"source": "test1.pdf"}), 0.8),
        (Mock(page_content="Test document 2", metadata={"source": "test2.pdf"}), 0.7),
    ]
    return mock_vs


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    mock_emb = Mock(spec=HuggingFaceEmbeddings)
    mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock_emb


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Mock(
            page_content="Apple Inc. reported revenue of $5.2B in Q1 2024",
            metadata={
                "source": "AAPL_10K_2024.pdf",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "document_type": "10-K",
                "fiscal_year": 2024,
                "quarter": "Q1"
            }
        ),
        Mock(
            page_content="Microsoft's cloud revenue grew 15% year-over-year",
            metadata={
                "source": "MSFT_10Q_2024.pdf", 
                "company": "Microsoft Corporation",
                "ticker": "MSFT",
                "document_type": "10-Q",
                "fiscal_year": 2024,
                "quarter": "Q2"
            }
        )
    ]
