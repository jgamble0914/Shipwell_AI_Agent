# AI Document Q&A Agent

An autonomous AI agent built with LangChain that ingests documents in multiple formats and answers questions about their content using retrieval-augmented generation (RAG).

## Features

- **Multi-format Document Support**:

  - Text files (`.txt`)
  - PDF documents (`.pdf`)
  - Word documents (`.docx`, `.doc`)
  - Images (`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`)
  - Markdown files (`.md`, `.markdown`)
  - CSV files (`.csv`)

- **Intelligent Retrieval**:

  - Vector-based semantic search using ChromaDB
  - Configurable chunking and retrieval parameters
  - Context-aware question answering

- **Source Citations**:

  - Shows source documents for each answer
  - Displays retrieved context chunks
  - Tracks file metadata

- **Modular Architecture**:
  - Separated concerns (loaders, indexing, retrieval, answering)
  - Easy to extend and customize
  - Clear, well-documented code

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (required):

   Create a `.env` file in the project root:

   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   **Note**: OPENAI_API_KEY is required to run this application. The application uses OpenAI for both embeddings and LLM generation.

4. **Install Tesseract OCR** (for image processing):

   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Usage

### Basic Usage

1. **Prepare your documents**:
   Place your documents in a folder (e.g., `./sample_docs`)

2. **Run the application**:

   ```bash
   python app.py --docs ./sample_docs
   ```

3. **Ask questions**:
   Once the indexing is complete, you'll enter an interactive prompt loop. Type your questions and press Enter.

4. **Exit**:
   Type `quit` to exit the application.

### Command Line Options

```bash
python app.py --docs <folder_path> [options]

Required:
  --docs PATH          Path to folder containing documents

Optional:
  --reindex           Force reindexing even if vector store exists
  --chunk-size N      Size of text chunks (default: 1000)
  --chunk-overlap N   Overlap between chunks (default: 200)
  --k N               Number of documents to retrieve (default: 4)
```

### Examples

```bash
# Basic usage
python app.py --docs ./sample_docs

# Force reindexing
python app.py --docs ./sample_docs --reindex

# Custom chunking parameters
python app.py --docs ./sample_docs --chunk-size 1500 --chunk-overlap 300

# Retrieve more documents per query
python app.py --docs ./sample_docs --k 6
```

## Adding Documents

### Supported Formats

The agent automatically detects and processes the following file formats:

- **Text**: `.txt`
- **PDF**: `.pdf`
- **Word**: `.docx`, `.doc`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`
- **Markdown**: `.md`, `.markdown`
- **CSV**: `.csv`

### How to Add Documents

1. **Place files in your documents folder**:
   Simply copy your documents into the folder you specify with `--docs`

2. **Subdirectories are supported**:
   The agent recursively searches all subdirectories

3. **Reindex when needed**:
   - If you add new documents, use `--reindex` to rebuild the index
   - Or delete the `./chroma_db` folder and run again

## Design Notes & Trade-offs

### Architecture Decisions

1. **Modular Design**:

   - **Loaders** (`src/loaders.py`): Handles all document format loading
   - **Indexing** (`src/indexing.py`): Manages chunking and vector store creation
   - **Retrieval** (`src/retrieval.py`): Handles QA chain and response formatting
   - **Utils** (`src/utils.py`): Shared utility functions

   This separation makes it easy to:

   - Add new document formats (extend `loaders.py`)
   - Swap vector stores (modify `indexing.py`)
   - Change LLM providers (modify `retrieval.py`)

2. **Vector Store Choice**: ChromaDB

   - **Pros**: Easy to use, persistent storage, good performance
   - **Cons**: Less scalable than cloud solutions for very large datasets
   - **Alternative**: Could easily swap to Pinecone, Weaviate, or FAISS

3. **Embedding Models**:

   - **OpenAI embeddings**: Required for the application to function
   - Uses OpenAI's text-embedding models for semantic search

4. **Chunking Strategy**:

   - **RecursiveCharacterTextSplitter**: Handles various document structures
   - **Configurable size/overlap**: Allows tuning for different document types
   - **Trade-off**: Larger chunks = more context but less precise retrieval

5. **LLM Integration**:
   - **OpenAI GPT models**: Required for answer generation
   - Uses GPT-3.5-turbo by default (configurable)
   - **Future**: Could add support for other providers (Anthropic, etc.)

### Scalability Considerations

1. **Document Volume**:

   - Current design handles hundreds to thousands of documents well
   - For larger scale, consider:
     - Batch processing for indexing
     - Distributed vector stores (Pinecone, Weaviate)
     - Async document loading

2. **Model Swapping**:

   - Easy to swap embeddings: Change `_get_embeddings()` in `indexing.py`
   - Easy to swap LLMs: Change `_get_llm()` in `retrieval.py`
   - Easy to swap vector stores: Modify `indexing.py` initialization

3. **API Deployment**:
   - Current CLI can be wrapped in FastAPI/Flask
   - Vector store persistence allows stateless API design
   - Could add caching layer for common queries

### Extensibility

**Adding a New Document Format**:

1. Add file extension to `DocumentLoader` class
2. Add loader logic in `load_document()` method
3. Install required dependencies

**Adding a New Vector Store**:

1. Modify `DocumentIndexer` to use different vector store
2. Update `QARetriever` to work with new store interface

**Adding a New LLM Provider**:

1. Extend `_get_llm()` in `retrieval.py`
2. Ensure compatibility with LangChain's LLM interface

## Future Enhancements

Potential improvements for future versions:

- Web interface (Flask/FastAPI)
- Multi-language support
- Advanced citation formatting (page numbers, sections)
- Query history and conversation context
- Support for more document formats (Excel, PowerPoint)
- Streaming responses for long answers
- Confidence scores for answers
- Batch question processing
