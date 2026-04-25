from __future__ import annotations

from typing import Any, Iterable

from elasticsearch import Elasticsearch, helpers


class ElasticIndex:
    def __init__(self, client: Elasticsearch, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    def exists(self) -> bool:
        return self.client.indices.exists(index=self.index_name)

    def create(self, mappings: dict[str, Any] | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if mappings:
            body["mappings"] = mappings
        if settings:
            body["settings"] = settings

        return self.client.indices.create(
            index=self.index_name,
            **body,
        )

    def create_if_missing(
        self,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        if not self.exists():
            self.create(mappings=mappings, settings=settings)

    def index_document(self, document: dict[str, Any], doc_id: str | None = None, refresh: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "index": self.index_name,
            "document": document,
        }
        if doc_id is not None:
            kwargs["id"] = doc_id
        if refresh:
            kwargs["refresh"] = "true"

        return self.client.index(**kwargs)

    def bulk_index(
        self,
        documents: Iterable[dict[str, Any]],
        id_field: str | None = None,
    ) -> tuple[int, list[Any]]:
        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_source": doc,
            }
            if id_field and id_field in doc:
                action["_id"] = str(doc[id_field])
            actions.append(action)

        return helpers.bulk(self.client, actions)

    def search(self, query: dict[str, Any], size: int = 10) -> dict[str, Any]:
        return self.client.search(
            index=self.index_name,
            query=query,
            size=size,
        )