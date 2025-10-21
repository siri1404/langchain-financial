"""Query preprocessing for financial document retrieval.

This module handles query preprocessing including entity extraction,
number normalization, and query expansion for financial terms.
"""

import re
from typing import Dict, Any, List, Optional


class QueryPreprocessor:
    """Preprocess financial queries for better retrieval.

    Features:
    - Entity extraction (companies, tickers, dates, financial terms)
    - Number normalization ($5.2M → 5200000)
    - Query expansion (revenue → top line, sales)
    - Financial term recognition

    Example:
        >>> preprocessor = QueryPreprocessor()
        >>> result = preprocessor.preprocess("What is AAPL revenue in Q4 2024?")
        >>> print(result["entities"])
        {'ticker': ['AAPL'], 'quarter': ['Q4'], 'year': ['2024']}
    """

    def __init__(
        self,
        extract_entities: bool = True,
        normalize_numbers: bool = True,
        expand_synonyms: bool = True,
        financial_entity_patterns: Optional[Dict[str, str]] = None,
    ):
        """Initialize the query preprocessor.

        Args:
            extract_entities: Whether to extract financial entities from query
            normalize_numbers: Whether to normalize number formats
            expand_synonyms: Whether to expand financial synonyms
            financial_entity_patterns: Custom regex patterns for entity extraction
        """
        self.extract_entities = extract_entities
        self.normalize_numbers = normalize_numbers
        self.expand_synonyms = expand_synonyms

        # Financial entity patterns
        self.patterns = financial_entity_patterns or {
            "ticker": r"\b(?:AAPL|MSFT|GOOGL|AMZN|TSLA|META|NVDA|NFLX|ADBE|CRM|ORCL|INTC|CSCO|IBM|AMD|QCOM|AVGO|TXN|AMAT|MU|LRCX|KLAC|MCHP|ADI|SNPS|CDNS|ANSS|FTNT|PANW|CRWD|ZS|OKTA|NET|DDOG|SNOW|PLTR|RBLX|U|ROKU|SPOT|SQ|PYPL|V|MA|AXP|COF|DFS|WU|FIS|FISV|GPN|FLT|VRSK|MCO|SPGI|NDAQ|CME|ICE|CBOE|MKTX|NDAQ|TROW|BLK|SCHW|AMG|BEN|IVZ|JHG|LM|PGR|ALL|TRV|CB|AIG|HIG|PRU|MET|AFL|PFG|UNM|LNC|BRO|RE|RGA|WRB|AXS|RLI|SIGI|KMPR|CINF|FAF|AFG|THG|EMCI|EIG|KINS|UFCS|ACGL|ARGO|AMSF|CNA|GNW|HALL|KNSL|MCY|MTG|NAVG|NMIH|OSCR|PRA|SAFT|STFC|UVE|WTM|Y|ZEN|ZTS|ABT|JNJ|PFE|MRK|LLY|ABBV|TMO|DHR|BMY|AMGN|GILD|BIIB|REGN|VRTX|ILMN|MRNA|BNTX|NVAX|JAZZ|INCY|EXAS|SGEN|BMRN|ARCT|CRTX|FOLD|IONS|MRNA|NVAX|SGEN|VRTX|BIIB|REGN|GILD|AMGN|BMY|LLY|MRK|PFE|JNJ|ABT|TMO|DHR|ABBV|ZTS|Y|ZEN|WTM|UVE|STFC|SAFT|PRA|OSCR|NMIH|NAVG|MTG|MCY|KNSL|HALL|GNW|CNA|ARGO|ACGL|UFCS|KINS|EIG|EMCI|AFG|FAF|CINF|KMPR|SIGI|RLI|AXS|WRB|RGA|RE|BRO|LNC|UNM|AFL|PFG|MET|AIG|PRU|HIG|TRV|ALL|PGR|AMG|BEN|IVZ|JHG|LM|SCHW|BLK|TROW|NDAQ|ICE|CME|NDAQ|MKTX|CBOE|SPGI|MCO|VRSK|FLT|GPN|FISV|FIS|WU|COF|DFS|AXP|MA|V|PYPL|SQ|SPOT|ROKU|U|RBLX|PLTR|SNOW|DDOG|NET|OKTA|ZS|CRWD|PANW|FTNT|CDNS|SNPS|ADI|KLAC|MCHP|AMAT|MU|LRCX|QCOM|AMD|IBM|CSCO|INTC|ORCL|CRM|ADBE|NFLX|NVDA|META|TSLA|AMZN|GOOGL|MSFT|AAPL)\b",  # Common ticker symbols
            "year": r"\b(?:19|20)\d{2}\b",  # 2024, 2023
            "quarter": r"\bQ[1-4]\b",  # Q1, Q2, Q3, Q4
            "fiscal_year": r"\bFY\s*\d{2,4}\b",  # FY2024, FY24
            "money": r"\$[\d,]+(?:\.\d+)?[KMB]?",  # $5.2M, $100K
            "percentage": r"\d+(?:\.\d+)?%",  # 15.5%, 20%
            "date": r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        }

        # Financial synonyms for query expansion
        self.synonyms = {
            "revenue": ["sales", "top line", "total revenue", "net revenue", "turnover"],
            "profit": ["net income", "earnings", "bottom line", "net profit"],
            "ebitda": ["operating income", "operating profit", "EBIT"],
            "risk": ["risk factor", "threat", "challenge", "uncertainty"],
            "growth": ["increase", "expansion", "rise"],
            "decline": ["decrease", "reduction", "drop", "fall"],
            "margin": ["profit margin", "operating margin", "gross margin"],
            "debt": ["liabilities", "borrowing", "leverage"],
            "cash": ["liquidity", "cash flow", "working capital"],
        }

        # Common financial metrics
        self.financial_metrics = {
            "eps", "pe", "p/e", "roe", "roa", "roi", "ebitda", "gaap",
            "non-gaap", "fcf", "capex", "opex", "cogs", "sg&a", "r&d",
            "arr", "mrr", "ltv", "cac", "churn", "nps"
        }

    def preprocess(self, query: str) -> Dict[str, Any]:
        """Preprocess query and return structured info.

        Args:
            query: The search query string

        Returns:
            Dictionary containing:
                - original_query: Original query string
                - processed_query: Processed query string
                - entities: Extracted entities by type
                - normalized_numbers: Normalized number mappings
                - expanded_terms: Query expansion variants
                - detected_metrics: Financial metrics found in query

        Example:
            >>> preprocessor.preprocess("AAPL revenue $5.2M in Q4")
            {
                'original_query': 'AAPL revenue $5.2M in Q4',
                'processed_query': 'AAPL revenue 5200000 in Q4',
                'entities': {'ticker': ['AAPL'], 'quarter': ['Q4'], 'money': ['$5.2M']},
                'normalized_numbers': {'$5.2M': 5200000.0},
                'expanded_terms': [...],
                'detected_metrics': ['revenue']
            }
        """
        result = {
            "original_query": query,
            "processed_query": query,
            "entities": {},
            "normalized_numbers": {},
            "expanded_terms": [],
            "detected_metrics": [],
        }

        # Extract entities
        if self.extract_entities:
            result["entities"] = self._extract_entities(query)

        # Normalize numbers
        if self.normalize_numbers:
            query, normalized = self._normalize_numbers(query)
            result["normalized_numbers"] = normalized
            result["processed_query"] = query

        # Expand synonyms
        if self.expand_synonyms:
            result["expanded_terms"] = self._expand_query(query)

        # Detect financial metrics
        result["detected_metrics"] = self._detect_financial_metrics(query)

        return result

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract financial entities from query.

        Args:
            query: Query string

        Returns:
            Dictionary mapping entity types to lists of extracted entities
        """
        entities = {}
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                # Deduplicate while preserving order
                entities[entity_type] = list(dict.fromkeys(matches))
        return entities

    def _normalize_numbers(self, query: str) -> tuple[str, Dict[str, float]]:
        """Normalize financial numbers: $5.2M → 5200000.

        Args:
            query: Query string containing financial numbers

        Returns:
            Tuple of (normalized_query, mappings_dict)
        """
        normalized = {}

        def replace_money(match):
            value = match.group(0)
            # Remove $ and commas, extract numeric part
            numeric_str = value.replace("$", "").replace(",", "")

            # Check for K/M/B suffix
            multiplier = 1
            if numeric_str.endswith("K"):
                multiplier = 1_000
                numeric_str = numeric_str[:-1]
            elif numeric_str.endswith("M"):
                multiplier = 1_000_000
                numeric_str = numeric_str[:-1]
            elif numeric_str.endswith("B"):
                multiplier = 1_000_000_000
                numeric_str = numeric_str[:-1]

            try:
                number = float(numeric_str) * multiplier
                normalized[value] = number
                return str(int(number))
            except ValueError:
                return value

        # Replace money patterns
        processed = re.sub(r"\$[\d,]+(?:\.\d+)?[KMB]?", replace_money, query)
        return processed, normalized

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with financial synonyms.

        Args:
            query: Original query

        Returns:
            List of expanded query variants
        """
        expanded = []
        query_lower = query.lower()

        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                for syn in synonyms:
                    expanded_query = re.sub(
                        rf"\b{term}\b",
                        syn,
                        query_lower,
                        flags=re.IGNORECASE
                    )
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)

        return expanded

    def _detect_financial_metrics(self, query: str) -> List[str]:
        """Detect financial metrics in query.

        Args:
            query: Query string

        Returns:
            List of detected financial metrics
        """
        query_lower = query.lower()
        detected = []

        for metric in self.financial_metrics:
            # Match whole words or common patterns
            pattern = rf"\b{re.escape(metric)}\b"
            if re.search(pattern, query_lower):
                detected.append(metric)

        # Also check synonyms keys
        for term in self.synonyms.keys():
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, query_lower):
                detected.append(term)

        return list(set(detected))
