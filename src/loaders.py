"""Document loaders for various file formats."""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_core.documents import Document


class DocumentLoader:
    """Handles loading documents from various formats."""

    # Supported file extensions
    TEXT_EXTENSIONS = {".txt"}
    PDF_EXTENSIONS = {".pdf"}
    DOCX_EXTENSIONS = {".docx", ".doc"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    MARKDOWN_EXTENSIONS = {".md", ".markdown"}
    CSV_EXTENSIONS = {".csv"}

    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """
        Load a single document based on its file extension.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects with content and metadata
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        # Add file path to metadata
        metadata = {"source": str(file_path), "file_name": file_path.name}

        try:
            if extension in cls.TEXT_EXTENSIONS:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

            elif extension in cls.PDF_EXTENSIONS:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

            elif extension in cls.DOCX_EXTENSIONS:
                loader = UnstructuredWordDocumentLoader(str(file_path))
                docs = loader.load()

            elif extension in cls.IMAGE_EXTENSIONS:
                loader = UnstructuredImageLoader(str(file_path))
                docs = loader.load()

            elif extension in cls.MARKDOWN_EXTENSIONS:
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()

            elif extension in cls.CSV_EXTENSIONS:
                loader = CSVLoader(str(file_path))
                docs = loader.load()

            else:
                print(f"Warning: Unsupported file format: {extension}")
                return []

            # Add metadata to all documents
            for doc in docs:
                doc.metadata.update(metadata)
                doc.metadata["file_type"] = extension[1:]  # Remove the dot

            return docs

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return []

    @classmethod
    def load_folder(cls, folder_path: str) -> List[Document]:
        """
        Load all supported documents from a folder.

        Args:
            folder_path: Path to the folder containing documents

        Returns:
            List of all Document objects from the folder
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        all_documents = []
        supported_extensions = (
            cls.TEXT_EXTENSIONS
            | cls.PDF_EXTENSIONS
            | cls.DOCX_EXTENSIONS
            | cls.IMAGE_EXTENSIONS
            | cls.MARKDOWN_EXTENSIONS
            | cls.CSV_EXTENSIONS
        )

        # Recursively find all supported files
        for file_path in folder_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                print(f"Loading: {file_path.name}")
                docs = cls.load_document(str(file_path))
                all_documents.extend(docs)

        print(f"\nLoaded {len(all_documents)} document chunks from {folder_path}")
        return all_documents
