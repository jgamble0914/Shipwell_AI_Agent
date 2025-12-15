"""Document indexing and vector store management."""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os


class DocumentIndexer:
    """Handles document chunking and vector store indexing."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: Optional[str] = None,
        persist_directory: str = "./chroma_db",
    ):
        """
        Initialize the document indexer.

        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model: Name of embedding model (OpenAI only, requires OPENAI_API_KEY)
            persist_directory: Directory to persist vector store
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        self.embeddings = self._get_embeddings(embedding_model)
        self.vector_store = None

    def _get_embeddings(self, model_name: Optional[str] = None):
        """Get OpenAI embedding model. Requires OPENAI_API_KEY to be set."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in your .env file or environment variables."
            )

        try:
            return OpenAIEmbeddings()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenAI embeddings: {e}. "
                "Please check your OPENAI_API_KEY is valid."
            ) from e

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better retrieval.

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []

        print(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        return chunks

    def index_documents(
        self, documents: List[Document], collection_name: str = "documents"
    ) -> Chroma:
        """
        Index documents into a vector store.

        Args:
            documents: List of Document objects to index
            collection_name: Name of the collection in the vector store

        Returns:
            Chroma vector store instance
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        chunks = self.chunk_documents(documents)

        print(f"Indexing {len(chunks)} chunks into vector store...")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name,
        )

        print(f"Indexing complete! Vector store saved to {self.persist_directory}")
        return self.vector_store

    def load_existing_index(self, collection_name: str = "documents") -> Chroma:
        """
        Load an existing vector store from disk.

        Args:
            collection_name: Name of the collection to load

        Returns:
            Chroma vector store instance
        """
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Vector store not found at {self.persist_directory}")

        print(f"Loading existing vector store from {self.persist_directory}...")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        return self.vector_store
