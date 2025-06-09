from loguru import logger
from .chromadb import get_chromadb_retriever
from .model import AgentState
from .util import get_google_llm
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser


def handle_rag_call(state: AgentState):
    logger.debug("handle rag call")
    question = state["messages"][0]
    logger.debug(f"{question=}")

    retriever = get_chromadb_retriever()
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"""
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    llm = get_google_llm()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke(question)
    logger.debug(f"response from rag call: {result}")
    return {"messages": [result]}


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
