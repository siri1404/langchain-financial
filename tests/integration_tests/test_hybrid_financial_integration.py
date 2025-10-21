"""Integration tests for hybrid financial retriever."""

import pytest
from unittest.mock import Mock, patch
from langchain_financial.hybrid_financial import HybridFinancialRetriever


class TestHybridFinancialRetrieverIntegration:
    """Integration test cases for HybridFinancialRetriever."""
    
    @patch('langchain_financial.hybrid_financial.ReciprocalRankFusion')
    @patch('langchain_financial.hybrid_financial.FinancialQueryPreprocessor')
    def test_end_to_end_retrieval(self, mock_preprocessor, mock_fusion):
        """Test end-to-end retrieval workflow."""
        # Setup mocks
        mock_vectorstore = Mock()
        mock_docs = [
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
        
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.9),
            (mock_docs[1], 0.8),
        ]
        
        mock_preprocessor_instance = Mock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_preprocessor_instance.preprocess.return_value = "Apple Q1 2024 revenue"
        
        mock_fusion_instance = Mock()
        mock_fusion.return_value = mock_fusion_instance
        mock_fusion_instance.fuse.return_value = mock_docs
        
        # Create retriever
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            preprocess_query=True
        )
        
        # Test retrieval
        result = retriever.invoke("Apple's Q1 2024 revenue")
        
        # Assertions
        assert len(result) == 2
        assert result[0].page_content == "Apple Inc. reported revenue of $5.2B in Q1 2024"
        assert result[0].metadata["company"] == "Apple Inc."
        assert result[0].metadata["ticker"] == "AAPL"
        
        # Verify preprocessing was called
        mock_preprocessor_instance.preprocess.assert_called_once_with("Apple's Q1 2024 revenue")
        
        # Verify vectorstore search was called
        mock_vectorstore.similarity_search_with_score.assert_called_once()
    
    def test_retrieval_with_metadata_filtering(self):
        """Test retrieval with metadata filtering."""
        mock_vectorstore = Mock()
        mock_docs = [
            Mock(
                page_content="Apple Inc. revenue data",
                metadata={
                    "company": "Apple Inc.",
                    "document_type": "10-K",
                    "fiscal_year": 2024
                }
            ),
            Mock(
                page_content="Microsoft revenue data", 
                metadata={
                    "company": "Microsoft Corporation",
                    "document_type": "10-Q",
                    "fiscal_year": 2024
                }
            )
        ]
        
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.9),
            (mock_docs[1], 0.8),
        ]
        
        # Create retriever with filters
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            company="Apple Inc.",
            document_type="10-K"
        )
        
        result = retriever.invoke("revenue data")
        
        # Should only return Apple's 10-K document
        assert len(result) == 1
        assert result[0].metadata["company"] == "Apple Inc."
        assert result[0].metadata["document_type"] == "10-K"
    
    def test_retrieval_with_multiple_filters(self):
        """Test retrieval with multiple metadata filters."""
        mock_vectorstore = Mock()
        mock_docs = [
            Mock(
                page_content="Apple Q1 2024 data",
                metadata={
                    "company": "Apple Inc.",
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "quarter": "Q1"
                }
            ),
            Mock(
                page_content="Apple Q2 2024 data",
                metadata={
                    "company": "Apple Inc.", 
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "quarter": "Q2"
                }
            )
        ]
        
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.9),
            (mock_docs[1], 0.8),
        ]
        
        # Create retriever with multiple filters
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            company="Apple Inc.",
            quarter="Q1"
        )
        
        result = retriever.invoke("Apple Q1 data")
        
        # Should only return Q1 data
        assert len(result) == 1
        assert result[0].metadata["quarter"] == "Q1"
        assert result[0].metadata["company"] == "Apple Inc."
    
    def test_retrieval_with_empty_results(self):
        """Test retrieval when no documents match filters."""
        mock_vectorstore = Mock()
        mock_docs = [
            Mock(
                page_content="Microsoft data",
                metadata={
                    "company": "Microsoft Corporation",
                    "document_type": "10-Q"
                }
            )
        ]
        
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.9),
        ]
        
        # Create retriever with filters that won't match
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            company="Apple Inc.",  # Won't match Microsoft
            document_type="10-K"   # Won't match 10-Q
        )
        
        result = retriever.invoke("test query")
        
        # Should return empty results
        assert len(result) == 0
    
    def test_retrieval_with_custom_weights(self):
        """Test retrieval with custom dense/sparse weights."""
        mock_vectorstore = Mock()
        mock_docs = [
            Mock(page_content="Test doc 1", metadata={"id": 1}),
            Mock(page_content="Test doc 2", metadata={"id": 2}),
        ]
        
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.9),
            (mock_docs[1], 0.8),
        ]
        
        # Create retriever with custom weights
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            dense_weight=0.8,
            sparse_weight=0.2
        )
        
        result = retriever.invoke("test query")
        
        # Should return results (weights don't affect basic functionality)
        assert len(result) == 2
    
    def test_retrieval_with_preprocessing_disabled(self):
        """Test retrieval with query preprocessing disabled."""
        mock_vectorstore = Mock()
        mock_docs = [Mock(page_content="Test doc", metadata={"id": 1})]
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.9),
        ]
        
        retriever = HybridFinancialRetriever(
            vectorstore=mock_vectorstore,
            k=10,
            preprocess_query=False
        )
        
        original_query = "Apple's Q1 2024 revenue"
        result = retriever.invoke(original_query)
        
        # Verify the original query was used (not preprocessed)
        call_args = mock_vectorstore.similarity_search_with_score.call_args[0]
        assert call_args[0] == original_query
        
        assert len(result) == 1
