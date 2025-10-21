"""Example usage of HybridFinancialRetriever.

This script demonstrates how to use the retriever with financial documents.
"""

from langchain_core.documents import Document
from langchain_financial import HybridFinancialRetriever


def create_sample_documents() -> list[Document]:
    """Create sample financial documents for testing."""
    return [
        Document(
            page_content="""
            Risk Factors

            Our business is subject to various risks including market volatility,
            regulatory changes, and competitive pressures. We face significant
            competition from both established players and new entrants in the
            technology sector. Economic downturns could materially affect our
            revenue and profitability.
            """,
            metadata={
                "source": "AAPL_10K_2024.pdf",
                "document_type": "10-K",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "section": "Risk Factors",
                "page": 15,
            },
        ),
        Document(
            page_content="""
            Management's Discussion and Analysis

            Revenue increased 15% year-over-year to $394.3 billion, driven by
            strong iPhone and Services performance. Gross margin expanded 150
            basis points to 43.5%. Operating expenses as a percentage of revenue
            decreased to 9.5% from 10.2% in the prior year.
            """,
            metadata={
                "source": "AAPL_10K_2024.pdf",
                "document_type": "10-K",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "section": "MD&A",
                "page": 28,
            },
        ),
        Document(
            page_content="""
            Quarterly Results

            For the quarter ended December 31, 2024, we reported revenue of
            $119.6 billion, up 11% year-over-year. iPhone revenue was $69.7
            billion, Services revenue reached $23.1 billion, and Mac revenue
            was $10.4 billion. Earnings per diluted share were $2.18, up 13%
            from Q4 2023.
            """,
            metadata={
                "source": "AAPL_10Q_Q4_2024.pdf",
                "document_type": "10-Q",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "quarter": "Q4",
                "section": "Financial Results",
                "page": 5,
            },
        ),
        Document(
            page_content="""
            CEO Tim Cook: Thank you for joining us today. We're pleased to
            report record quarterly revenue driven by strong demand across all
            product categories. Our Services business continues to show remarkable
            strength, growing 16% year-over-year. Looking ahead, we remain focused
            on innovation and delivering exceptional value to our customers.
            """,
            metadata={
                "source": "AAPL_Earnings_Call_Q4_2024.txt",
                "document_type": "earnings_call",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "quarter": "Q4",
                "section": "CEO Remarks",
                "speaker": "Tim Cook",
            },
        ),
        Document(
            page_content="""
            Risk Factors

            We are subject to intense competition in the cloud computing market.
            Amazon Web Services faces competition from Microsoft Azure, Google Cloud,
            and other providers. Pricing pressure could impact our margins. We also
            face risks related to data security, privacy regulations, and potential
            service disruptions.
            """,
            metadata={
                "source": "AMZN_10K_2024.pdf",
                "document_type": "10-K",
                "company": "Amazon.com Inc.",
                "ticker": "AMZN",
                "fiscal_year": 2024,
                "section": "Risk Factors",
                "page": 12,
            },
        ),
        Document(
            page_content="""
            Financial Performance

            Net sales increased 12% to $574.8 billion for fiscal year 2024.
            North America segment sales were $352.8 billion, International
            segment sales were $131.2 billion, and AWS sales were $90.8 billion,
            growing 13% year-over-year. Operating income increased to $36.9 billion.
            """,
            metadata={
                "source": "AMZN_10K_2024.pdf",
                "document_type": "10-K",
                "company": "Amazon.com Inc.",
                "ticker": "AMZN",
                "fiscal_year": 2024,
                "section": "MD&A",
                "page": 25,
            },
        ),
    ]


def example_basic_usage():
    """Example 1: Basic usage with minimal configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    # Note: In real usage, you would use a proper vectorstore
    # For this example, we'll demonstrate the API
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import FakeEmbeddings

        # Create sample documents
        docs = create_sample_documents()

        # Create vector store
        embeddings = FakeEmbeddings(size=768)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Create retriever with basic config
        retriever = HybridFinancialRetriever(
            vectorstore=vectorstore,
            documents=docs,  # For BM25 sparse retrieval
            k=3,
        )

        # Query
        query = "What are Apple's risk factors?"
        print(f"\nQuery: {query}")
        print(f"\nRetrieving {retriever.k} documents...\n")

        results = retriever.invoke(query)

        # Display results
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Section: {doc.metadata.get('section')}")
            print(f"Company: {doc.metadata.get('company')}")
            if "rrf_score" in doc.metadata:
                print(f"RRF Score: {doc.metadata['rrf_score']:.4f}")
            print(f"\nContent preview:")
            print(doc.page_content[:200].strip() + "...")

    except ImportError as e:
        print(f"Skipping example (missing dependencies): {e}")


def example_with_filtering():
    """Example 2: Using metadata filters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: With Metadata Filtering")
    print("=" * 70)

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import FakeEmbeddings

        docs = create_sample_documents()
        embeddings = FakeEmbeddings(size=768)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Create retriever with filters
        retriever = HybridFinancialRetriever(
            vectorstore=vectorstore,
            documents=docs,
            k=5,
            # Metadata filters
            document_types=["10-K", "10-Q"],  # Only annual and quarterly reports
            companies=["Apple"],  # Only Apple documents
            fiscal_years=[2024],  # Only 2024 filings
        )

        query = "revenue growth"
        print(f"\nQuery: {query}")
        print(f"Filters: document_types=['10-K', '10-Q'], companies=['Apple'], fiscal_years=[2024]")
        print(f"\nRetrieving documents...\n")

        results = retriever.invoke(query)

        print(f"Found {len(results)} documents matching filters:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc.metadata.get('document_type')} - {doc.metadata.get('section')}")
            print(f"   Company: {doc.metadata.get('company')}, Year: {doc.metadata.get('fiscal_year')}")

    except ImportError as e:
        print(f"Skipping example (missing dependencies): {e}")


def example_advanced_config():
    """Example 3: Advanced configuration with custom weights."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Advanced Configuration")
    print("=" * 70)

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import FakeEmbeddings

        docs = create_sample_documents()
        embeddings = FakeEmbeddings(size=768)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Create retriever with custom weights
        retriever = HybridFinancialRetriever(
            vectorstore=vectorstore,
            documents=docs,
            k=3,
            # Custom fusion weights
            dense_weight=0.7,  # Favor semantic search
            sparse_weight=0.3,  # Less weight on keyword matching
            # Preprocessing options
            normalize_numbers=True,
            extract_entities=True,
            expand_synonyms=False,
        )

        # Query with numbers and entities
        query = "What is AAPL revenue in 2024?"
        print(f"\nQuery: {query}")
        print(f"Config: dense_weight=0.7, sparse_weight=0.3")
        print(f"Preprocessing: normalize_numbers=True, extract_entities=True\n")

        results = retriever.invoke(query)

        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Document: {doc.metadata.get('document_type')} - {doc.metadata.get('section')}")
            if "rrf_score" in doc.metadata:
                print(f"RRF Score: {doc.metadata['rrf_score']:.4f}")
            print(f"Content: {doc.page_content[:150].strip()}...")

    except ImportError as e:
        print(f"Skipping example (missing dependencies): {e}")


def example_async_usage():
    """Example 4: Async usage for better performance."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Async Usage")
    print("=" * 70)

    import asyncio

    async def async_retrieve():
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import FakeEmbeddings

            docs = create_sample_documents()
            embeddings = FakeEmbeddings(size=768)
            vectorstore = FAISS.from_documents(docs, embeddings)

            retriever = HybridFinancialRetriever(
                vectorstore=vectorstore,
                documents=docs,
                k=3,
            )

            # Async single query
            query = "cloud computing risks"
            print(f"\nAsync Query: {query}\n")

            results = await retriever.ainvoke(query)

            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.metadata.get('company')} - {doc.metadata.get('section')}")

            # Async batch queries
            print("\n\nAsync Batch Queries:")
            queries = [
                "What are the risk factors?",
                "revenue growth trends",
                "AWS performance",
            ]

            batch_results = await retriever.abatch(queries)

            for query, docs in zip(queries, batch_results):
                print(f"\n'{query}' -> {len(docs)} documents")

        except ImportError as e:
            print(f"Skipping example (missing dependencies): {e}")

    # Run async example
    asyncio.run(async_retrieve())


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("HYBRID FINANCIAL RETRIEVER - USAGE EXAMPLES")
    print("=" * 70)

    example_basic_usage()
    example_with_filtering()
    example_advanced_config()
    example_async_usage()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
