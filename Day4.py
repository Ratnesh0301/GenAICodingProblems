from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import Literal, TypedDict
from langchain_openai import ChatOpenAI

class RAGState(TypedDict):
    question: str
    documents: list
    generation: str
    grade: str # "relevant" or "irrelevant"
    iteration: int

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

def retrieve(state: RAGState)-> dict:
    print("---RETRIEVE---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "iterations": state.get("iteration",0)+1}

def grade_documents(state:RAGState)-> dict:
    print("---GRADING---")
    question, docs = state['question'], state['documents']

    grade_prompt = f""" Question: {question}

    Document: {docs[0].page_content if docs else 'none'}

    Is this document relevant to the question? 
    Return only 'yes' or 'no'
    """

    response = llm.invoke([HumanMessage(grade_prompt)])

    grade = 'relevant' if 'yes' in response.content.lower() else 'irrelevant'
    
    return {'grade':grade}

def web_search_fallback(state: RAGState) -> dict:
    print("--- WEB FALLBACK ---")
    # In production: use TavilySearchResults or SerperAPI
    fallback_doc = type('Doc', (), {'page_content': f'Web context for: {state["question"]}'})()
    return {"documents": [fallback_doc]}


def generate(state: RAGState) -> dict:
    print("--- GENERATE ---")
    context = "\n".join(d.page_content for d in state["documents"])
    prompt = f"Context: {context}\n\nQuestion: {state['question']}\n\nAnswer:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": response.content}

def route_after_grade(state: RAGState) -> Literal["web_search_fallback", "generate"]:
    if state["grade"] == "irrelevant" and state["iterations"] < 2:
        return "web_search_fallback"
    return "generate"

#Build graph
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search_fallback", web_search_fallback)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", route_after_grade)
workflow.add_edge("web_search_fallback", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()
result = graph.invoke({"question": "What are PCI-DSS encryption requirements?"})
    


