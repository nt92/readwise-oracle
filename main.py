import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def find_similar():
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="data",
        )
    )
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/gtr-t5-large"
    )
    collection = client.get_collection(
        name="highlights",
        embedding_function=embedding_function,
    )
    while True:
        query = input('Enter a query: ')
        results = collection.query(
            query_texts=[query],
            n_results=10,
            include=["documents", "distances"]
        )
        for distance, document in zip(results['distances'][0], results['documents'][0]):
            print(f"{distance} : {document}")


if __name__ == '__main__':
    find_similar()
