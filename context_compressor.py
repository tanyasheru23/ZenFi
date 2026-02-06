from difflib import SequenceMatcher


SIMILARITY_THRESHOLD = 0.75


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compress_context(docs):
    """
    Remove highly similar chunks before sending to LLM.
    """

    filtered_docs = []

    for doc in docs:

        is_duplicate = False

        for existing in filtered_docs:
            score = similarity(
                doc.page_content,
                existing.page_content
            )

            if score > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_docs.append(doc)

    return filtered_docs
