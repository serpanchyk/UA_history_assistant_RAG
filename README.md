# Ukrainian History Student Assistant

## System Pipeline (How it Works)

The project operates in three main steps controlled by the provided scripts:

### Step 1: Ingestion & Processing (`main.py`)
* **Extraction:** Parses PDF textbooks to separate text and images.
* **Filtering:** Removes non-relevant images (QR codes, UI elements, blurred placeholders).
* **Linking:** Uses spatial algorithms to link images to their nearest text captions on the page.
* **Chunking:** Splits text into semantic chunks (~1500 characters) with overlap.

### Step 2: Indexing (Vector Storage)
* **Embedding:**
    * **Text:** Converted into hybrid vectors (Dense + Sparse) using `BAAI/bge-m3`.
    * **Images:** Converted into semantic vectors using `multilingual-clip`.
* **Storage:** All vectors and metadata are indexed into **Qdrant**.

### Step 3: Retrieval & Serving (`app.py`)
* **RAG:** Retrieves relevant text and images based on the user's query.
* **Generation:** Uses `GPT-4o` to generate a grounded answer containing the retrieved visual context.

---

## Setup & Installation

### Prerequisites
* **Docker** & **Docker Compose**
* **NVIDIA GPU** + **NVIDIA Container Toolkit** (Recommended)
* **Azure OpenAI Access** (API Key & Endpoint)

### 1. Clone the Repository
```bash
git clone [https://github.com/your_username/UA_history_assistant_RAG](https://github.com/your_username/UA_history_assistant_RAG)
cd UA_history_assistant_RAG
```

Create a .env file in the project root:
```
# Azure OpenAI Config
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=[https://your-resource.openai.azure.com/](https://your-resource.openai.azure.com/)
AZURE_OPENAI_API_VERSION=2024-05-01-preview

# Qdrant Config
QDRANT_URL=http://localhost:6333  # Use 'qdrant_storage' if inside Docker network
QDRANT_API_KEY=

# Project Settings
RAG_DATA_DIR=./data
```

---

Run via Docker (Recommended)
```bash
docker-compose up --build
```

Open your browser to: http://localhost:7860

---

Run loc