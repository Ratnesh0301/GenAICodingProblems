"""
Coding problem: LCEL RAG chain with structured output
med
Build a production-style RAG chain using LCEL pipes with Pydantic output validation — the modern LangChain pattern.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

class ComplianceResult(BaseModel):
    status: Literal['compliant','violation','needs_review']
    severity: Literal['low','medium','high','critical']
    finding: str = Field(description="Specific finding or observation")
    recommendation: str = Field(description="Actionable recommendation")
    confidence: float = Field(description="Confidence score 0.0-1.0")

def build_compliance_chain(retriever):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ('system',"You are an IT compliance expert. Assess the architecture"
                   "against retrieved policy context. Be precise and actionable."),
        ("human","Architecture Description: {question}\n\n"
                  "Policy Context:\n{context}\n\n"
                  "Return a structured compliance assessment.")
    ])

    #Structured LLM Output
    structured_llm = llm.with_structured_output(ComplianceResult)

    #LCEL Pipeline: Fetch Context + question in parallel, then generate
    chain = (
        RunnableParallel({
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        })
        | prompt
        | structured_llm
    )

    return chain


chain = build_compliance_chain(retriever)
result = chain.invoke("The API Gateway lacks mTLS between microservices.")
print(result.status, result.severity, result.recommendation)
