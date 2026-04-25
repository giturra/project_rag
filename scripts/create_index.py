import argparse
from typing import Iterable, Iterator, TypeVar

from elasticsearch import Elasticsearch

from src.core.mappings import DOCUMENT_MAPPING
from src.search.index import ElasticIndex
from src.utils.json_io import iter_jsonl


T = TypeVar("T")


def limit_iter(items: Iterable[T], limit: int | None) -> Iterator[T]:
    if limit is None:
        yield from items
        return

    if limit < 0:
        raise ValueError("limit must be >= 0")

    for i, item in enumerate(items):
        if i >= limit:
            break
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index JSONL documents into Elasticsearch"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:9200",
        help="Elasticsearch host",
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Elasticsearch index name",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL file",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="id",
        help="Field to use as Elasticsearch document ID",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of documents per Elasticsearch bulk request",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to index, useful for testing",
    )

    args = parser.parse_args()

    es = Elasticsearch(args.host)

    index = ElasticIndex(
        client=es,
        index_name=args.index,
    )

    index.create_if_missing(mappings=DOCUMENT_MAPPING)

    docs = iter_jsonl(args.input)
    docs = limit_iter(docs, args.limit)

    success_count, errors = index.bulk_index(
        documents=docs,
        id_field=args.id_field,
        chunk_size=args.chunk_size,
    )

    print(f"Indexed documents: {success_count}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()