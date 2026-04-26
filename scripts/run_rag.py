from src.search.sparse import SparseRetriever

def main():
    retriever = ElasticsearchBM25Retriever()
    query = "What is the capital of France?"
    docs = retriever.retrieve(query)
    print(docs)