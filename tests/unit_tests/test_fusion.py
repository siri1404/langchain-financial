"""Unit tests for fusion module."""

import pytest
from unittest.mock import Mock
from langchain_financial.fusion import ReciprocalRankFusion


class TestReciprocalRankFusion:
    """Test cases for ReciprocalRankFusion."""
    
    def test_init(self):
        """Test fusion initialization."""
        fusion = ReciprocalRankFusion()
        assert fusion.k == 60  # Default k value
    
    def test_init_custom_k(self):
        """Test fusion initialization with custom k."""
        fusion = ReciprocalRankFusion(k=100)
        assert fusion.k == 100
    
    def test_fuse_empty_lists(self):
        """Test fusing empty lists."""
        fusion = ReciprocalRankFusion()
        result = fusion.fuse([])
        assert result == []
    
    def test_fuse_single_list(self):
        """Test fusing a single list."""
        fusion = ReciprocalRankFusion()
        docs = [
            Mock(page_content="doc1", metadata={"id": 1}),
            Mock(page_content="doc2", metadata={"id": 2}),
        ]
        result = fusion.fuse([docs])
        assert len(result) == 2
        assert result[0].metadata["id"] == 1
        assert result[1].metadata["id"] == 2
    
    def test_fuse_multiple_lists(self):
        """Test fusing multiple lists."""
        fusion = ReciprocalRankFusion()
        
        docs1 = [
            Mock(page_content="doc1", metadata={"id": 1}),
            Mock(page_content="doc2", metadata={"id": 2}),
        ]
        docs2 = [
            Mock(page_content="doc2", metadata={"id": 2}),
            Mock(page_content="doc3", metadata={"id": 3}),
        ]
        
        result = fusion.fuse([docs1, docs2])
        assert len(result) == 3  # Should have 3 unique documents
        assert result[0].metadata["id"] == 2  # doc2 should be ranked highest
        assert result[1].metadata["id"] == 1  # doc1 should be second
        assert result[2].metadata["id"] == 3  # doc3 should be third
    
    def test_fuse_with_scores(self):
        """Test fusing lists with scores."""
        fusion = ReciprocalRankFusion()
        
        docs1 = [
            (Mock(page_content="doc1", metadata={"id": 1}), 0.9),
            (Mock(page_content="doc2", metadata={"id": 2}), 0.8),
        ]
        docs2 = [
            (Mock(page_content="doc2", metadata={"id": 2}), 0.7),
            (Mock(page_content="doc3", metadata={"id": 3}), 0.6),
        ]
        
        result = fusion.fuse([docs1, docs2])
        assert len(result) == 3
        # doc2 should be ranked highest due to appearing in both lists
        assert result[0].metadata["id"] == 2
    
    def test_fuse_different_lengths(self):
        """Test fusing lists of different lengths."""
        fusion = ReciprocalRankFusion()
        
        docs1 = [Mock(page_content="doc1", metadata={"id": 1})]
        docs2 = [
            Mock(page_content="doc2", metadata={"id": 2}),
            Mock(page_content="doc3", metadata={"id": 3}),
            Mock(page_content="doc4", metadata={"id": 4}),
        ]
        
        result = fusion.fuse([docs1, docs2])
        assert len(result) == 4
        # All documents should be present
        ids = [doc.metadata["id"] for doc in result]
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids
        assert 4 in ids
