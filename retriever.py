import re
from collections import defaultdict


def is_numeric_query(query):
    """
    Detect finance/numeric questions.
    """
    return bool(re.search(r"\d|%|rupee|salary|tax|amount", query.lower()))


def diversified_retrieval(vectordb, query, k=10):

    # Step 1: retrieve more candidates
    docs = vectordb.similarity_search(query, k=k)

    grouped = defaultdict(list)

    # Step 2: group by source
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped[source].append(doc)

    final_docs = []

    numeric_query = is_numeric_query(query)

    # Step 3: select balanced chunks
    for source, source_docs in grouped.items():

        # prefer table chunks for numeric queries
        if numeric_query:
            source_docs.sort(
                key=lambda d: 0 if d.metadata.get("type") == "table" else 1
            )

        final_docs.extend(source_docs[:2])  # max 2 per source

    # Step 4: limit total context
    return final_docs[:5]
