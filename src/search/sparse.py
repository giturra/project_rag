"""Sparse BM25 retriever using Elasticsearch."""

from __future__ import annotations

from typing import Any, List

from elasticsearch import Elasticsearch

from src.core.schema import Query, ScoredDocument
from src.core.ranking import Ranking
from src.search.retriever import Retriever


class ElasticsearchBM25Retriever(Retriever):
    def __init__(
        self,
        client: Elasticsearch,
        index_name: str,
        fields: list[str] | None = None,
    ) -> None:
        super().__init__(index_name=index_name)
        self.client = client
        self.fields = fields or ["title", "text"]

    def retrieve(self, query: Query, num_results: int = 1000) -> Ranking:
        result = self.client.search(
            index=self._index_name,
            query={
                "multi_match": {
                    "query": query.question,
                    "fields": self.fields,
                }
            },
            size=num_results,
        )

        return self.process_results(query, result)

    def batch_query(
        self,
        queries: List[Query],
        top_k: int = 1000,
    ) -> list[dict[str, Any]]:
        results = []

        for query in queries:
            result = self.client.search(
                index=self._index_name,
                query={
                    "multi_match": {
                        "query": query.question,
                        "fields": self.fields,
                    }
                },
                size=top_k,
            )
            results.append(result)

        return results

    def process_results(self, query: Query, results: dict[str, Any]) -> Ranking:
        scored_documents = []

        for hit in results["hits"]["hits"]:
            source = hit.get("_source", {})

            scored_documents.append(
                ScoredDocument(
                    doc_id=str(source.get("id", hit["_id"])),
                    content=source.get("text", ""),
                    score=float(hit["_score"] or 0.0),
                )
            )

        return Ranking(
            query_id=query.query_id,
            scored_docs=scored_documents,
        )