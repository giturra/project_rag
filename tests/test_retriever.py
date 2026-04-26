from elasticsearch import Elasticsearch

from src.core.schema import Query
from src.search.sparse import ElasticsearchBM25Retriever


def main() -> None:
    # Config (edit here)
    host = "http://localhost:9200"
    index_name = "dpr_wiki"
    query_text = "Who was Aaron in the Bible?"
    top_k = 5

    # Setup
    es = Elasticsearch(host)

    retriever = ElasticsearchBM25Retriever(
        client=es,
        index_name=index_name,
    )

    query = Query(
        query_id="test-query",
        question=query_text,
    )

    # Retrieve
    ranking = retriever.retrieve(
        query=query,
        num_results=top_k,
    )

    # Print results
    print(f"\nQuery: {query.question}")
    print(f"Top {top_k} results:\n")

    idxs, docs = ranking.documents()

    for i, doc in zip(idxs, docs):


        print(f"doc_id={i} score={doc}")
    
        break


if __name__ == "__main__":
    main()