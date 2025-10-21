"""Unit tests for preprocessing module."""

import pytest
from unittest.mock import Mock
from langchain_financial.preprocessing import FinancialQueryPreprocessor


class TestFinancialQueryPreprocessor:
    """Test cases for FinancialQueryPreprocessor."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = FinancialQueryPreprocessor()
        assert preprocessor.normalize_numbers is True
        assert preprocessor.extract_entities is True
    
    def test_init_custom_params(self):
        """Test preprocessor initialization with custom parameters."""
        preprocessor = FinancialQueryPreprocessor(
            normalize_numbers=False,
            extract_entities=False
        )
        assert preprocessor.normalize_numbers is False
        assert preprocessor.extract_entities is False
    
    def test_preprocess_basic_query(self):
        """Test preprocessing a basic query."""
        preprocessor = FinancialQueryPreprocessor()
        query = "What is Apple's revenue?"
        result = preprocessor.preprocess(query)
        assert result == "What is Apple's revenue?"
    
    def test_preprocess_with_numbers(self):
        """Test preprocessing with number normalization."""
        preprocessor = FinancialQueryPreprocessor()
        query = "Show me companies with revenue over $5.2M"
        result = preprocessor.preprocess(query)
        # Should normalize the number format
        assert "$5.2M" in result or "5200000" in result
    
    def test_preprocess_with_entities(self):
        """Test preprocessing with entity extraction."""
        preprocessor = FinancialQueryPreprocessor()
        query = "What are Apple's risk factors in Q1 2024?"
        result = preprocessor.preprocess(query)
        assert "Apple" in result
        assert "Q1" in result
        assert "2024" in result
    
    def test_preprocess_ticker_extraction(self):
        """Test ticker symbol extraction."""
        preprocessor = FinancialQueryPreprocessor()
        query = "How is AAPL performing?"
        result = preprocessor.preprocess(query)
        assert "AAPL" in result
    
    def test_preprocess_date_extraction(self):
        """Test date extraction."""
        preprocessor = FinancialQueryPreprocessor()
        query = "Show earnings for 2024-03-15"
        result = preprocessor.preprocess(query)
        assert "2024-03-15" in result
    
    def test_preprocess_quarter_extraction(self):
        """Test quarter extraction."""
        preprocessor = FinancialQueryPreprocessor()
        query = "Q3 2024 financial results"
        result = preprocessor.preprocess(query)
        assert "Q3" in result
        assert "2024" in result
    
    def test_preprocess_without_normalization(self):
        """Test preprocessing without number normalization."""
        preprocessor = FinancialQueryPreprocessor(normalize_numbers=False)
        query = "Revenue over $5.2M"
        result = preprocessor.preprocess(query)
        assert "$5.2M" in result
    
    def test_preprocess_without_entities(self):
        """Test preprocessing without entity extraction."""
        preprocessor = FinancialQueryPreprocessor(extract_entities=False)
        query = "Apple's Q1 2024 results"
        result = preprocessor.preprocess(query)
        # Should not extract entities
        assert result == "Apple's Q1 2024 results"
    
    def test_preprocess_empty_query(self):
        """Test preprocessing empty query."""
        preprocessor = FinancialQueryPreprocessor()
        result = preprocessor.preprocess("")
        assert result == ""
    
    def test_preprocess_none_query(self):
        """Test preprocessing None query."""
        preprocessor = FinancialQueryPreprocessor()
        result = preprocessor.preprocess(None)
        assert result is None
