"""Query and Document classes as representation of query and document in a
collection respectively."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Query:
    """Representation of a query.

    It contains query ID and question."""

    query_id: str
    question: str

    def __str__(self):
        return self.question

    @staticmethod
    def load_queries_from_file(filename: str) -> List[Query]:
        """Extracts a list of Query objects from CSV file.

        Args:
            filename: The name of the file.

        Returns:
            Extracted Queries.
        """
        file_content = pd.read_csv(filename, sep="\t", header=None)
        queries = []
        # for _, row in file_content.iterrows():
        #     queries.append(Query(query_id=row["id"], question=row["query"]))
        for _, (id, question) in file_content.iterrows():
            queries.append(Query(query_id=id, question=question))

        return queries


@dataclass
class Document:
    """Representation of a document. It contains doc_id and optionally
    document content."""

    doc_id: str
    content: str = None


@dataclass
class ScoredDocument(Document):
    """Representation of a retrieved document. It contains doc_id and optionally
    document content and ranking score."""

    doc_id: str
    score: float = 0