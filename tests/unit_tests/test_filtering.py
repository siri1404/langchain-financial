"""Unit tests for filtering module."""

import pytest
from unittest.mock import Mock
from langchain_financial.filtering import FinancialMetadataFilter


class TestFinancialMetadataFilter:
    """Test cases for FinancialMetadataFilter."""
    
    def test_init(self):
        """Test filter initialization."""
        filter_obj = FinancialMetadataFilter()
        assert filter_obj.filters == {}
    
    def test_add_filter(self):
        """Test adding filters."""
        filter_obj = FinancialMetadataFilter()
        filter_obj.add_filter("company", "Apple Inc.")
        assert filter_obj.filters["company"] == "Apple Inc."
    
    def test_add_multiple_filters(self):
        """Test adding multiple filters."""
        filter_obj = FinancialMetadataFilter()
        filter_obj.add_filter("company", "Apple Inc.")
        filter_obj.add_filter("document_type", "10-K")
        assert len(filter_obj.filters) == 2
        assert filter_obj.filters["company"] == "Apple Inc."
        assert filter_obj.filters["document_type"] == "10-K"
    
    def test_clear_filters(self):
        """Test clearing all filters."""
        filter_obj = FinancialMetadataFilter()
        filter_obj.add_filter("company", "Apple Inc.")
        filter_obj.clear_filters()
        assert filter_obj.filters == {}
    
    def test_apply_filters_match(self):
        """Test applying filters when document matches."""
        filter_obj = FinancialMetadataFilter()
        filter_obj.add_filter("company", "Apple Inc.")
        
        doc = Mock()
        doc.metadata = {"company": "Apple Inc.", "ticker": "AAPL"}
        
        result = filter_obj.apply_filters(doc)
        assert result is True
    
    def test_apply_filters_no_match(self):
        """Test applying filters when document doesn't match."""
        filter_obj = FinancialMetadataFilter()
        filter_obj.add_filter("company", "Apple Inc.")
        
        doc = Mock()
        doc.metadata = {"company": "Microsoft Corporation", "ticker": "MSFT"}
        
        result = filter_obj.apply_filters(doc)
        assert result is False
    
    def test_apply_filters_partial_match(self):
        """Test applying filters with partial matches."""
        filter_obj = FinancialMetadataFilter()
        filter_obj.add_filter("company", "Apple")
        
        doc = Mock()
        doc.metadata = {"company": "Apple Inc.", "ticker": "AAPL"}
        
        result = filter_obj.apply_filters(doc)
        assert result is True
    
    def test_apply_filters_missing_metadata(self):
        """Test applying filters when metadata is missing."""
        filter_obj = FinancialMetadataFilter()
        filter_obj.add_filter("company", "Apple Inc.")
        
        doc = Mock()
        doc.metadata = {"ticker": "AAPL"}  # Missing company field
        
        result = filter_obj.apply_filters(doc)
        assert result is False
    
    def test_apply_filters_empty_filters(self):
        """Test applying empty filters (should match all)."""
        filter_obj = FinancialMetadataFilter()
        
        doc = Mock()
        doc.metadata = {"company": "Apple Inc."}
        
        result = filter_obj.apply_filters(doc)
        assert result is True
