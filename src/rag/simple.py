from base import RAGBase


class SimpleRAG(RAGBase):
    def __init__(self, retriever, generator):
        super().__init__(retriever, generator)
    
    def generate(self, query: str, docs):
        context = "\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.llm.generate([{"role": "user", "content": prompt}]) 