from abc import ABC, abstractmethod

class BaseRAG(ABC):
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def run(self, query: str):
        query = self.preprocess(query)
        docs = self.retrieve(query)
        return self.generate(query, docs)

    def preprocess(self, query: str) -> str:
        return query 

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)

    @abstractmethod
    def generate(self, query: str, docs):
        pass