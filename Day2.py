"""Coding problem: Hybrid search with ChromaDB + BM25
med
Implement hybrid retrieval combining semantic search and BM25, then fuse with RRF. Core pattern for production RAG."""

from transformers.models.mra.modeling_mra import sparse_dense_mm
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_classic.schema import Document

embeddings = OpenAIEmbeddings()

def build_hybrid_retriever(docs: list[Document], k: int = 5):
    """
    Production hybrid retriever: semantic + BM25 fused via RRF
    """

    #Dense Retriever
    vectorstore = Chroma.from_documents(docs, embeddings)
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20}
    )

    #Sparse Retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    #Emsemble with RRF Fusion (Weighted Sum to 1)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever,bm25_retriever],
        weights=[0.5,0.5],
    )

    return hybrid_retriever