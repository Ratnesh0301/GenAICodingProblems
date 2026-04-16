"""
Coding problem: ChromaDB with metadata filtering + Qdrant comparison
Implement production-style ChromaDB with metadata filters, then show equivalent Qdrant pattern. Key for multi-standard compliance systems.
"""

from langchain_classic.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.schema import Document

embeddings = OpenAIEmbeddings()

#----ChromaDB with Metadata Filtering----

def build_compliance_vectorstore(docs: list[Document]):
    """
    Each doc should have metadata:
    {standard: "PCI-DSS", section:"3.4", severity:"critical"}
    """
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="compliance_policies",
        persist_directory="./chroma_db" #Persistance for Production
    )

def retrieve_by_standard(vectorstore: Chroma, query: str,
                        standard: str, k: int = 5):
    """
    Metadata pre-filter: only search within one compliance standard"""

    return vectorstore.similarity_search(
        query=query,
        k=k,
        filter={"standard": standard} # prefilter reduces search space
    )

#-----Qdrant Equivalent (Production Pattern)--------------

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def qdrant_filtered_search(client: QdrantClient, query_vector: list,
                                standard: str, k: int = 5):
    """
    Qdrant support complex payload filters with AND/OR logic
    """

    return client.search(
        collection_name="compliance_policies",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="standard",
                    match=MatchValue(value=standard)
                )
            ]               
        ),
        limit=k
    )

# Interview point: "We chose ChromaDB for dev simplicity but the 
# architecture is designed to swap to Qdrant for production scale
# — the LangChain VectorStore interface abstracts the backend."  

