import csv

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def get_highlights(csv_file):
    readwise_highlights = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            readwise_highlights.append(row['Highlight'])
    return readwise_highlights


chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="data"
))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/gtr-t5-large"
)
highlights = get_highlights('data/readwise-highlights.csv')
collection = chroma_client.create_collection(
    name="highlights",
    embedding_function=embedding_function,
)
collection.add(
    documents=highlights,
    ids=[str(i) for i in range(len(highlights))],
)
