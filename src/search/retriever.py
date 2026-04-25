"""Interface for first pass retrieval."""

from abc import ABC, abstractmethod
from src.core.schema import Query
from src.core.ranking import Ranking
from typing import List


class Retriever(ABC):
    def __init__(self, index_name: str) -> None:
        """Abstract class for first-pass retrieval.

        Args:
            index_name: Index name.
        """
        self._index_name = index_name

    @abstractmethod
    def retrieve(self, query: Query, num_results: int = 1000) -> Ranking:
        """Interface for first-pass retrieval that needs to be implemented.

        Args:
            query: Query instance.
            num_results: Number of results to return (defaults
                to 1000).

        Raises:
            NotImplementedError: Raised if the method is not overwritten.

        Returns:
            Ranking of documents.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_query(
        self, queries: List[Query], top_k: int = 1000
    ) -> list[dict]:
        """Batch query an index and return the results.

        Args:
            queries: List of queries to embed.
            top_k: Number of results to return. Defaults to 1000.

        Raises:
            NotImplementedError: Raised if the method is not overwritten.

        Returns:
            A list of dictionaries with the results for each query.
        """
        raise NotImplementedError

    @abstractmethod
    def process_results(self, query: Query, results) -> Ranking:
        """Process the results from the index and return a Ranking object.

        Args:
            query: Query object.
            results: Results from Pinecone.

        Raises:
            NotImplementedError: Raised if the method is not overwritten.

        Returns:
            A Ranking object with the results.
        """
        raise NotImplementedError

    def batch_retrieve(
        self, queries: List[Query], num_results: int = 1000
    ) -> List[Ranking]:
        """Retrieves documents for a list of queries in a batch.

        Args:
            queries: List of Query objects.
            num_results: Number of results to return for each query.

        Returns:
            A list of Ranking objects with the retrieved documents for each
            query in a batch.
        """
        batch_results = self.batch_query(queries, num_results)

        rankings = []
        for query, results in zip(queries, batch_results):
            rankings.append(self.process_results(query, results))

        return rankings