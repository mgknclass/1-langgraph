from loguru import logger
from .config import settings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_embeddings() -> Embeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


def get_text_documents():
    loader = DirectoryLoader(
        path=settings.data_path, glob="./*.txt", loader_cls=TextLoader
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return text_splitter.split_documents(documents=docs)


def get_chromadb_retriever() -> VectorStoreRetriever:
    db = Chroma.from_documents(
        documents=get_text_documents(), embedding=get_embeddings()
    )
    return db.as_retriever(search_kwargs={"k": 3})
