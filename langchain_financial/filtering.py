"""Metadata filtering for financial documents.

This module provides filtering capabilities for financial document metadata
including document type, company, fiscal period, and section.
"""

from typing import List, Optional, Callable, Any
from langchain_core.documents import Document


class MetadataFilter:
    """Filter documents by financial metadata.

    Supports filtering by:
    - Document type (10-K, 10-Q, 8-K, earnings_call)
    - Company name or ticker
    - Fiscal year, quarter
    - Section (Risk Factors, MD&A, Financial Statements)
    - Filing date ranges
    - Custom filter functions

    Example:
        >>> filter = MetadataFilter(
        ...     document_types=["10-K", "10-Q"],
        ...     companies=["AAPL", "Apple Inc."],
        ...     fiscal_years=[2024, 2023]
        ... )
        >>> filtered = filter.filter(documents)
    """

    def __init__(
        self,
        document_types: Optional[List[str]] = None,
        companies: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        fiscal_years: Optional[List[int]] = None,
        quarters: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        custom_filters: Optional[List[Callable[[Document], bool]]] = None,
    ):
        """Initialize the metadata filter.

        Args:
            document_types: Filter by document type (e.g., ["10-K", "10-Q"])
            companies: Filter by company name (case-insensitive partial match)
            tickers: Filter by ticker symbol (case-insensitive)
            fiscal_years: Filter by fiscal year (e.g., [2024, 2023])
            quarters: Filter by quarter (e.g., ["Q1", "Q2"])
            sections: Filter by section name (e.g., ["Risk Factors"])
            min_confidence: Minimum confidence score (if available)
            custom_filters: List of custom filter functions
                Each function takes a Document and returns bool
        """
        self.document_types = [dt.upper() for dt in document_types] if document_types else None
        self.companies = [c.lower() for c in companies] if companies else None
        self.tickers = [t.upper() for t in tickers] if tickers else None
        self.fiscal_years = fiscal_years
        self.quarters = [q.upper() for q in quarters] if quarters else None
        self.sections = sections
        self.min_confidence = min_confidence
        self.custom_filters = custom_filters or []

    def filter(self, documents: List[Document]) -> List[Document]:
        """Apply all filters to document list.

        Filters are applied in sequence. A document must pass ALL filters
        to be included in results.

        Args:
            documents: List of documents to filter

        Returns:
            Filtered list of documents

        Example:
            >>> docs = [doc1, doc2, doc3]
            >>> filtered = metadata_filter.filter(docs)
            >>> # Only docs matching all criteria
        """
        filtered = documents

        if self.document_types:
            filtered = self._filter_by_doc_type(filtered)

        if self.companies:
            filtered = self._filter_by_company(filtered)

        if self.tickers:
            filtered = self._filter_by_ticker(filtered)

        if self.fiscal_years:
            filtered = self._filter_by_year(filtered)

        if self.quarters:
            filtered = self._filter_by_quarter(filtered)

        if self.sections:
            filtered = self._filter_by_section(filtered)

        if self.min_confidence is not None:
            filtered = self._filter_by_confidence(filtered)

        # Apply custom filters
        for custom_filter in self.custom_filters:
            filtered = [doc for doc in filtered if custom_filter(doc)]

        return filtered

    def _filter_by_doc_type(self, docs: List[Document]) -> List[Document]:
        """Filter by document type.

        Checks metadata keys: document_type, doc_type, type

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents
        """
        return [
            doc for doc in docs
            if self._get_doc_type(doc) in self.document_types
        ]

    def _get_doc_type(self, doc: Document) -> str:
        """Get document type from metadata, checking multiple keys."""
        doc_type = (
            doc.metadata.get("document_type") or
            doc.metadata.get("doc_type") or
            doc.metadata.get("type") or
            ""
        )
        return doc_type.upper()

    def _filter_by_company(self, docs: List[Document]) -> List[Document]:
        """Filter by company name (case-insensitive partial match).

        Checks metadata keys: company, company_name

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents
        """
        return [
            doc for doc in docs
            if self._matches_company(doc)
        ]

    def _matches_company(self, doc: Document) -> bool:
        """Check if document matches any of the specified companies."""
        company = (
            doc.metadata.get("company") or
            doc.metadata.get("company_name") or
            ""
        )
        company_lower = company.lower()

        # Check if any specified company is in the document's company field
        return any(
            comp in company_lower
            for comp in self.companies
        )

    def _filter_by_ticker(self, docs: List[Document]) -> List[Document]:
        """Filter by ticker symbol (case-insensitive exact match).

        Checks metadata keys: ticker, symbol, stock_symbol

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents
        """
        return [
            doc for doc in docs
            if self._get_ticker(doc) in self.tickers
        ]

    def _get_ticker(self, doc: Document) -> str:
        """Get ticker from metadata, checking multiple keys."""
        ticker = (
            doc.metadata.get("ticker") or
            doc.metadata.get("symbol") or
            doc.metadata.get("stock_symbol") or
            ""
        )
        return ticker.upper()

    def _filter_by_year(self, docs: List[Document]) -> List[Document]:
        """Filter by fiscal year.

        Checks metadata keys: fiscal_year, year

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents
        """
        return [
            doc for doc in docs
            if self._get_fiscal_year(doc) in self.fiscal_years
        ]

    def _get_fiscal_year(self, doc: Document) -> Optional[int]:
        """Get fiscal year from metadata, checking multiple keys."""
        year = (
            doc.metadata.get("fiscal_year") or
            doc.metadata.get("year")
        )
        if year is not None:
            try:
                return int(year)
            except (ValueError, TypeError):
                return None
        return None

    def _filter_by_quarter(self, docs: List[Document]) -> List[Document]:
        """Filter by quarter.

        Checks metadata keys: quarter, fiscal_quarter

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents
        """
        return [
            doc for doc in docs
            if self._get_quarter(doc) in self.quarters
        ]

    def _get_quarter(self, doc: Document) -> str:
        """Get quarter from metadata, checking multiple keys."""
        quarter = (
            doc.metadata.get("quarter") or
            doc.metadata.get("fiscal_quarter") or
            ""
        )
        return str(quarter).upper()

    def _filter_by_section(self, docs: List[Document]) -> List[Document]:
        """Filter by document section.

        Checks metadata keys: section, section_name

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents
        """
        return [
            doc for doc in docs
            if self._get_section(doc) in self.sections
        ]

    def _get_section(self, doc: Document) -> str:
        """Get section from metadata, checking multiple keys."""
        return (
            doc.metadata.get("section") or
            doc.metadata.get("section_name") or
            ""
        )

    def _filter_by_confidence(self, docs: List[Document]) -> List[Document]:
        """Filter by minimum confidence score.

        Checks multiple confidence score keys and uses the highest available.

        Args:
            docs: Documents to filter

        Returns:
            Filtered documents meeting minimum confidence
        """
        return [
            doc for doc in docs
            if self._get_confidence(doc) >= self.min_confidence
        ]

    def _get_confidence(self, doc: Document) -> float:
        """Get confidence score from metadata.

        Checks multiple keys: confidence, rerank_score, rrf_score, score
        Returns the highest available score.
        """
        scores = []

        # Check various confidence score keys
        for key in ["confidence", "rerank_score", "rrf_score", "score"]:
            if key in doc.metadata:
                try:
                    scores.append(float(doc.metadata[key]))
                except (ValueError, TypeError):
                    pass

        return max(scores) if scores else 0.0

    def get_filter_stats(self, documents: List[Document]) -> dict[str, Any]:
        """Get statistics about what would be filtered.

        Useful for debugging filter effectiveness.

        Args:
            documents: Documents to analyze

        Returns:
            Dictionary with filter statistics:
                - total_docs: Total number of documents
                - filtered_docs: Number after filtering
                - filtered_by_type: Dict of document types and counts
                - filtered_by_year: Dict of years and counts
                - etc.

        Example:
            >>> stats = filter.get_filter_stats(documents)
            >>> print(f"Kept {stats['filtered_docs']} of {stats['total_docs']}")
        """
        stats = {
            "total_docs": len(documents),
            "filtered_docs": 0,
            "doc_types": {},
            "companies": {},
            "fiscal_years": {},
            "quarters": {},
            "sections": {},
        }

        filtered = self.filter(documents)
        stats["filtered_docs"] = len(filtered)

        # Count distributions in filtered results
        for doc in filtered:
            # Document types
            doc_type = self._get_doc_type(doc)
            stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1

            # Companies
            company = doc.metadata.get("company", "Unknown")
            stats["companies"][company] = stats["companies"].get(company, 0) + 1

            # Fiscal years
            year = self._get_fiscal_year(doc)
            if year:
                stats["fiscal_years"][year] = stats["fiscal_years"].get(year, 0) + 1

            # Quarters
            quarter = self._get_quarter(doc)
            if quarter:
                stats["quarters"][quarter] = stats["quarters"].get(quarter, 0) + 1

            # Sections
            section = self._get_section(doc)
            if section:
                stats["sections"][section] = stats["sections"].get(section, 0) + 1

        return stats
