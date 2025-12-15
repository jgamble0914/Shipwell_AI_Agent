"""Main CLI application for the AI Document Q&A Agent."""

import argparse
import sys
import os
from pathlib import Path
from src.loaders import DocumentLoader
from src.indexing import DocumentIndexer
from src.retrieval import QARetriever
from src.utils import load_env_file, validate_folder


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="AI Agent for Document Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --docs ./sample_docs
  python app.py --docs ./sample_docs --reindex
        """,
    )

    parser.add_argument(
        "--docs", type=str, required=True, help="Path to folder containing documents"
    )

    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing even if vector store exists",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)",
    )

    parser.add_argument(
        "--k", type=int, default=4, help="Number of documents to retrieve (default: 4)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_env_file()

    # Validate OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required.")
        print("Please set it in your .env file or as an environment variable.")
        print("Get your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)

    # Validate folder path
    if not validate_folder(args.docs):
        print(f"Error: Folder not found: {args.docs}")
        sys.exit(1)

    print("=" * 60)
    print("AI Document Q&A Agent")
    print("=" * 60)
    print()

    # Step 1: Load documents
    print("Step 1: Loading documents...")
    try:
        documents = DocumentLoader.load_folder(args.docs)
        if not documents:
            print("Error: No documents found in the specified folder.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)

    # Step 2: Index documents
    print("\nStep 2: Indexing documents...")
    indexer = DocumentIndexer(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    try:
        # Check if index exists
        if not args.reindex and Path(indexer.persist_directory).exists():
            print("Found existing index. Loading...")
            vector_store = indexer.load_existing_index()
        else:
            print("Creating new index...")
            vector_store = indexer.index_documents(documents)
    except Exception as e:
        print(f"Error indexing documents: {e}")
        sys.exit(1)

    # Step 3: Initialize QA retriever
    print("\nStep 3: Initializing QA system...")
    try:
        qa_retriever = QARetriever(vector_store, k=args.k)
    except Exception as e:
        print(f"Error initializing QA system: {e}")
        sys.exit(1)

    # Step 4: Interactive Q&A loop
    print("\n" + "=" * 60)
    print("Ready! Ask questions about your documents.")
    print("Type 'quit' to exit.")
    print("Type 'reindex' to reload and reindex documents from the folder.")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            # Handle special commands
            if question.lower() == "quit":
                print("\nGoodbye!")
                break

            if question.lower() == "reindex":
                print("\n" + "=" * 60)
                print("Reindexing documents...")
                print("=" * 60)

                try:
                    # Reload documents from folder
                    print("\nStep 1: Reloading documents from folder...")
                    documents = DocumentLoader.load_folder(args.docs)
                    if not documents:
                        print("Warning: No documents found in the folder.")
                        continue

                    # Reindex documents
                    print("\nStep 2: Reindexing documents...")
                    vector_store = indexer.index_documents(documents)

                    # Update QA retriever with new vector store
                    print("\nStep 3: Updating QA system...")
                    qa_retriever = QARetriever(vector_store, k=args.k)

                    print("\n" + "=" * 60)
                    print("Reindexing complete! The database has been updated.")
                    print("=" * 60)
                    print()

                except Exception as e:
                    print(f"\nError during reindexing: {e}")
                    print("The previous index is still available.")
                continue

            # Answer the question
            print("\nSearching documents...")
            result = qa_retriever.answer_question(question)

            # Display formatted response
            print("\n" + qa_retriever.format_response(result, show_context=True))

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
