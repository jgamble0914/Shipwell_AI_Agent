"""Retrieval and question-answering logic."""

from typing import Dict, Optional, Any
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os


class QARetriever:
    """Handles question answering with retrieval-augmented generation."""

    def __init__(
        self, vector_store: Chroma, llm_model: Optional[str] = None, k: int = 4
    ):
        """
        Initialize the QA retriever.

        Args:
            vector_store: Chroma vector store instance
            llm_model: Name of LLM model to use
            k: Number of documents to retrieve for each query
        """
        self.vector_store = vector_store
        self.k = k

        # Initialize LLM
        self.llm = self._get_llm(llm_model)

        # Create retrieval QA chain
        self.qa_chain = self._create_qa_chain()

    def _get_llm(self, model_name: Optional[str] = None):
        """Get OpenAI LLM. Requires OPENAI_API_KEY to be set."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in your .env file or environment variables."
            )

        try:
            model_name = model_name or "gpt-3.5-turbo"
            return ChatOpenAI(model=model_name, temperature=0)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenAI LLM: {e}. "
                "Please check your OPENAI_API_KEY is valid."
            ) from e

    def _create_qa_chain(self) -> Any:
        """Create the retrieval QA chain using LangChain API."""
        if self.llm is None:
            raise ValueError("LLM is not initialized. OPENAI_API_KEY is required.")

        # Create prompt template using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use the following pieces of context to answer the question at the end. "
                    "If you don't know the answer based on the context, just say that you don't know, "
                    "don't try to make up an answer.\n\nContext: {context}",
                ),
                ("human", "{input}"),
            ]
        )

        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
        qa_chain = create_retrieval_chain(retriever, document_chain)

        return qa_chain

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using retrieval-augmented generation.

        Args:
            question: User's question

        Returns:
            Dictionary with 'answer', 'sources', and 'context' keys
        """
        # Use QA chain
        result = self.qa_chain.invoke({"input": question})

        # Extract sources from retrieved documents
        source_docs = result.get("context", [])
        sources = list(
            set([doc.metadata.get("source", "Unknown") for doc in source_docs])
        )

        return {
            "answer": result.get("answer", ""),
            "sources": sources,
            "context": source_docs,
        }

    def format_response(self, result: Dict[str, Any], show_context: bool = True) -> str:
        """
        Format the QA response for display.

        Args:
            result: Result dictionary from answer_question
            show_context: Whether to show retrieved context chunks

        Returns:
            Formatted string response
        """
        output = []

        # Add answer
        output.append("=" * 60)
        output.append("ANSWER:")
        output.append("=" * 60)
        output.append(result["answer"])
        output.append("")

        # Add sources/citations
        if result["sources"]:
            output.append("=" * 60)
            output.append("SOURCES:")
            output.append("=" * 60)
            for i, source in enumerate(result["sources"], 1):
                output.append(f"{i}. {source}")
            output.append("")

        # Add context chunks if requested
        if show_context and result.get("context"):
            output.append("=" * 60)
            output.append("RETRIEVED CONTEXT:")
            output.append("=" * 60)
            for i, doc in enumerate(result["context"][:3], 1):  # Show top 3 chunks
                output.append(
                    f"\n[Chunk {i} from {doc.metadata.get('source', 'Unknown')}]"
                )
                output.append("-" * 60)
                # Truncate long chunks
                content = doc.page_content
                if len(content) > 300:
                    content = content[:300] + "..."
                output.append(content)

        return "\n".join(output)
