"""Unit tests for hybrid_financial module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_financial.hybrid_financial import HybridFinancialRetriever


class TestHybridFinancialRetriever:
    """Test cases for HybridFinancialRetriever."""
    
    def test_init(self):
        """Test retriever initialization."""
        mock_vectorstore = Mock()
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10
        )
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.k == 10
        assert retriever.dense_weight == 0.5
        assert retriever.sparse_weight == 0.5
    
    def test_init_custom_weights(self):
        """Test retriever initialization with custom weights."""
        mock_vectorstore = Mock()
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            dense_weight=0.7,
            sparse_weight=0.3
        )
        assert retriever.dense_weight == 0.7
        assert retriever.sparse_weight == 0.3
    
    def test_init_with_filters(self):
        """Test retriever initialization with filters."""
        mock_vectorstore = Mock()
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            company="Apple Inc."
        )
        assert retriever.filters["company"] == "Apple Inc."
    
    @patch('langchain_financial.hybrid_financial.ReciprocalRankFusion')
    def test_invoke_basic(self, mock_fusion):
        """Test basic invoke functionality."""
        mock_vectorstore = Mock()
        mock_docs = [
            Mock(page_content="doc1", metadata={"source": "test1.pdf"}),
            Mock(page_content="doc2", metadata={"source": "test2.pdf"}),
        ]
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.8),
            (mock_docs[1], 0.7),
        ]
        
        mock_fusion_instance = Mock()
        mock_fusion.return_value = mock_fusion_instance
        mock_fusion_instance.fuse.return_value = mock_docs
        
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10
        )
        
        result = retriever.invoke("test query")
        
        assert len(result) == 2
        assert result[0].page_content == "doc1"
        assert result[1].page_content == "doc2"
        mock_vectorstore.similarity_search_with_score.assert_called_once()
    
    def test_invoke_with_filters(self):
        """Test invoke with metadata filters."""
        mock_vectorstore = Mock()
        mock_docs = [
            Mock(page_content="doc1", metadata={"company": "Apple Inc."}),
            Mock(page_content="doc2", metadata={"company": "Microsoft Corp."}),
        ]
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.8),
            (mock_docs[1], 0.7),
        ]
        
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            company="Apple Inc."
        )
        
        result = retriever.invoke("test query")
        
        # Should filter out Microsoft doc
        assert len(result) == 1
        assert result[0].metadata["company"] == "Apple Inc."
    
    def test_invoke_empty_results(self):
        """Test invoke with empty results."""
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = []
        
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10
        )
        
        result = retriever.invoke("test query")
        assert result == []
    
    def test_invoke_with_preprocessing(self):
        """Test invoke with query preprocessing."""
        mock_vectorstore = Mock()
        mock_docs = [Mock(page_content="doc1", metadata={"source": "test1.pdf"})]
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.8),
        ]
        
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            preprocess_query=True
        )
        
        result = retriever.invoke("Apple's Q1 2024 revenue")
        
        # Should preprocess the query
        mock_vectorstore.similarity_search_with_score.assert_called_once()
        call_args = mock_vectorstore.similarity_search_with_score.call_args[0]
        assert "Apple" in call_args[0]  # Query should be preprocessed
    
    def test_add_filter(self):
        """Test adding filters."""
        mock_vectorstore = Mock()
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10
        )
        
        retriever.add_filter("document_type", "10-K")
        assert retriever.filters["document_type"] == "10-K"
    
    def test_clear_filters(self):
        """Test clearing filters."""
        mock_vectorstore = Mock()
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            company="Apple Inc."
        )
        
        retriever.clear_filters()
        assert retriever.filters == {}
    
    def test_get_relevant_documents(self):
        """Test get_relevant_documents method."""
        mock_vectorstore = Mock()
        mock_docs = [Mock(page_content="doc1", metadata={"source": "test1.pdf"})]
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.8),
        ]
        
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10
        )
        
        result = retriever.get_relevant_documents("test query")
        assert len(result) == 1
        assert result[0].page_content == "doc1"
