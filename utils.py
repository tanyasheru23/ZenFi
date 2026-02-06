TEMPLATE_KEYWORDS = [
    "example",
    "sample",
    "steps",
    "template",
    "illustration",
    "case",
    "plan",
    "allocation",
    "table",
    "checklist",
    "how to",
    "strategy"
]


def is_template_chunk(text):
    text = text.lower()
    return any(keyword in text for keyword in TEMPLATE_KEYWORDS)


def table_to_text(df):
    """
    Convert table dataframe into natural language sentences
    for better embeddings.
    """

    rows_text = []
    headers = list(df.columns)

    for _, row in df.iterrows():
        sentence_parts = []
        for h, val in zip(headers, row):
            sentence_parts.append(f"{h} is {val}")
        rows_text.append(", ".join(sentence_parts))

    return " ; ".join(rows_text)
