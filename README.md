# RAG Model

This project implements a Retrieval-Augmented Generation (RAG) system in Python. It leverages document retrieval and language generation to answer queries based on uploaded documents.

## Features
- Upload and process PDF documents
- Query documents using natural language
- Streamlit web interface for interaction
- Qdrant vector database integration (see `qdr.py`)

## Getting Started

### Prerequisites
- Python 3.12 or higher
- Required Python packages (see `requirement.txt`)

### Installation
1. Clone the repository or download the source code.
2. Install dependencies:
   ```powershell
   pip install -r requirement.txt
   ```

### Usage
- To run the Streamlit app:
  ```powershell
  streamlit run test.py
  ```
- To run the Qdrant integration script:
  ```powershell
  python qdr.py
  ```

### File Structure
- `main.py` - Main application logic
- `test.py` - Streamlit app entry point
- `qdr.py` - Qdrant vector database integration
- `uploads/` - Directory for uploaded documents
- `requirement.txt` - List of required Python packages
- `env.example` - Example environment variables file

## License
Specify your license here.

## Author
SyedAli-Kodexo
