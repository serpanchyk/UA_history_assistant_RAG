**[P1] Missing error handling for file I/O operations leads to silent failures**
- **Files:** `ingest/scrap_data.py`, `main.py`
- **Functions:** `scrap_text`, `scrap_images`, `question_llm`
- **Problem:** All file operations (CSV reads, PDF opens, file writes, text file reads) lack try-except blocks.  If `docs.csv` doesn't exist, PDFs are corrupted, or paths are invalid, the application will crash with unhandled exceptions, potentially losing data mid-processing.
- **Suggestion:** Wrap file operations in try-except blocks with proper logging: 
```
def scrap_text():
    try:
        df_docs = pd.read_csv(DFS_PATH / 'docs.csv', index_col='id')
    except FileNotFoundError:
        print(f"Error: {DFS_PATH / 'docs.csv'} not found")
        return
    except pd. errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return
    # ... rest of function
```

---

**[P1] Hardcoded relative paths break execution from different directories**
- **Files:** `ingest/scrap_data.py`, `main.py`, `rag/llm.py`
- **Functions:** `scrap_text`, `scrap_images`, `question_llm`, module-level in `llm.py`
- **Problem:** Paths like `'../data/pdfs'`, `'rag/prompt.txt'`, and `'models/MamayLM-Gemma-3-4B-IT-v1.0.Q4_K_S.gguf'` are relative to CWD.  Running scripts from different directories (e.g., from project root vs. ingest/ dir) will fail with FileNotFoundError.
- **Suggestion:** Use `__file__` to create absolute paths:
```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / 'data' / 'pdfs'
IMAGES_PATH = BASE_DIR / 'data' / 'images'
```

---

**[P1] Resource leaks:  PDF documents never closed**
- **Files:** `ingest/scrap_data.py`
- **Functions:** `scrap_text`, `scrap_images`
- **Problem:** `pymupdf.open(file_path)` opens file handles but never closes them. Processing multiple PDFs will exhaust file descriptors, causing crashes or memory leaks on large datasets.
- **Suggestion:** Use context managers: 
```
for index, doc_row in df_docs.iterrows():
    file_path = DOCS_PATH / doc_row['pdf_name']
    with pymupdf.open(file_path) as doc:
        for page in doc:
            # process pages
```

---

**[P2] Global LLM instance initialized at import time causes startup failures**
- **Files:** `rag/llm.py`
- **Functions:** `-` (module-level)
- **Problem:** The `llm` object is instantiated when the module is imported (line 3-9). If the model file doesn't exist or is corrupted, any import of this module crashes the entire application before code can handle it.  This also makes testing impossible without the 4GB+ model file.
- **Suggestion:** Use lazy initialization:
```
_llm = None
def get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(model_path="models/.. .", ...)
    return _llm

def get_response(content):
    llm = get_llm()
    # ... rest of function
```

---

**[P2] Missing CSV index column causes duplicate index issues**
- **Files:** `ingest/scrap_data.py`
- **Functions:** `scrap_text`, `scrap_images`
- **Problem:** Lines 42 and 81 save CSVs with `to_csv()` without `index=False`. This creates an unnamed index column.  Re-running the script and reading these CSVs will fail or produce incorrect data structures.
- **Suggestion:** Add `index=False` or explicitly name the index: 
```
df_text.to_csv(DFS_PATH / 'texts.csv', index=False)
df_images.to_csv(DFS_PATH / 'images. csv', index=False)
```

---

**[P2] DFS_PATH directory not created before use**
- **Files:** `ingest/scrap_data.py`
- **Functions:** `scrap_text`, `scrap_images`
- **Problem:** Line 8 creates `IMAGES_PATH`, but `DFS_PATH` is never created.  Calling `pd.read_csv(DFS_PATH / 'docs.csv')` or `to_csv(DFS_PATH / ...)` will fail with FileNotFoundError if the directory doesn't exist.
- **Suggestion:** Add directory creation:
```
DFS_PATH = Path('../data/dfs')
DFS_PATH.mkdir(parents=True, exist_ok=True)
```

---

**[P3] Magic number "type" values lack documentation and type safety**
- **Files:** `ingest/scrap_data.py`
- **Functions:** `scrap_text`, `scrap_images`
- **Problem:** Lines 26 (`if block['type'] == 1`) and 62 (`if block['type'] == 0`) use unexplained magic numbers. PyMuPDF uses 0=text, 1=image, but this is not documented in code. Reversing logic between functions is error-prone.
- **Suggestion:** Use constants: 
```
BLOCK_TYPE_TEXT = 0
BLOCK_TYPE_IMAGE = 1

# In scrap_text:
if block['type'] == BLOCK_TYPE_IMAGE:
    continue
# In scrap_images:
if block['type'] == BLOCK_TYPE_TEXT:
    continue
```

---

**[P3] Inefficient iterrows() usage with large datasets**
- **Files:** `ingest/scrap_data.py`
- **Functions:** `scrap_text`, `scrap_images`
- **Problem:** Lines 20 and 54 use `df_docs.iterrows()`, which is 10-100x slower than alternatives for large datasets. With 13+ textbooks, this becomes noticeable.
- **Suggestion:** Use `itertuples()` for better performance:
```
for doc in df_docs.itertuples():
    file_path = DOCS_PATH / doc.pdf_name
    # access other fields via doc.field_name
```

---

**[P3] No validation of required CSV columns**
- **Files:** `ingest/scrap_data. py`
- **Functions:** `scrap_text`, `scrap_images`
- **Problem:** Code assumes `docs.csv` has `pdf_name` column without validation. If the CSV is malformed or uses different column names, a KeyError will occur deep in the loop after partial processing.
- **Suggestion:** Validate schema upfront:
```
df_docs = pd.read_csv(DFS_PATH / 'docs.csv', index_col='id')
if 'pdf_name' not in df_docs.columns:
    raise ValueError("docs.csv missing required 'pdf_name' column")
```

---

**[P4] Missing function return values and status indicators**
- **Files:** `ingest/scrap_data.py`, `rag/llm.py`
- **Functions:** `scrap_text`, `scrap_images`, `get_response`
- **Problem:** All functions return `None` implicitly.  Callers have no way to know if operations succeeded, how many records were processed, or if errors occurred.  This violates error-reporting best practices.
- **Suggestion:** Return status information:
```
def scrap_text() -> int:
    # ... processing
    df_text. to_csv(DFS_PATH / 'texts.csv', index=False)
    return len(df_text)  # Return count of processed texts
```

---

**[P4] Typo in docstring: "texbook" should be "textbook"**
- **Files:** `ingest/scrap_data.py`
- **Functions:** `scrap_text`, `scrap_images`
- **Problem:** Lines 12 and 46 contain "texbook" instead of "textbook" in docstrings. This reduces code professionalism and searchability.
- **Suggestion:** Fix spelling:
```
"""
Takes textbook pdfs from docs dataset. 
```

---

**[P4] Encoding not specified for text file operations**
- **Files:** `main.py`
- **Functions:** `question_llm`
- **Problem:** Line 4 opens `rag/prompt.txt` without explicit encoding. Given the Ukrainian text content, this will fail on Windows (default cp1252) or other non-UTF-8 systems with UnicodeDecodeError.
- **Suggestion:** Always specify UTF-8:
```
with open('rag/prompt.txt', 'r', encoding='utf-8') as f:
    prompt = f.read()
```

---

**[P5] No logging framework - only print statements**
- **Files:** `rag/llm.py`
- **Functions:** `get_response`
- **Problem:** Lines 27 and 29 use `print()` for output.  This can't be controlled, redirected, or filtered by log level in production.  Debugging and monitoring become difficult.
- **Suggestion:** Use Python's logging module:
```
import logging
logger = logging.getLogger(__name__)

# Instead of print: 
logger.info(delta['content'])
```

---

**[P5] Missing __init__. py files for proper package structure**
- **Files:** `ingest/`, `rag/` directories
- **Functions:** `-`
- **Problem:** Based on the import structure (`from rag. llm import ... `), the directories should be Python packages but may lack `__init__.py` files. This can cause import errors in some Python versions or deployment scenarios.
- **Suggestion:** Add empty `__init__.py` files to `ingest/` and `rag/` directories, or use PEP 420 namespace packages explicitly. 

---

**[P6] No unit tests for critical data processing functions**
- **Files:** Project-wide
- **Functions:** `scrap_text`, `scrap_images`, `get_response`
- **Problem:** No test files exist in the repository.  The PDF scraping logic (block type filtering, bbox extraction, image saving) and LLM integration have no automated verification.  Regressions will go undetected.
- **Suggestion:** Create `tests/` directory with pytest tests:
```
# tests/test_scrap_data.py
def test_scrap_text_filters_images():
    # Mock pymupdf document with mixed blocks
    # Assert only text blocks are extracted
    pass
```

---

## Overall Summary

**Critical Risks:** The codebase has several **production-blocking issues**:  (1) unhandled file I/O exceptions will cause data loss and crashes, (2) relative path dependencies make the code non-portable across execution contexts, and (3) resource leaks from unclosed PDF handles will exhaust system resources on large datasets.  The global LLM initialization creates fragile imports that fail if the model is missing. 

**Estimated Rework Size:** Core issues require refactoring path handling, adding error boundaries, and fixing resource management, but the codebase is small and well-structured for remediation.

**Top 3 Actions to Merge Safely:**
1. **Add comprehensive error handling** around all file operations with try-except blocks and validation
2. **Fix path resolution** using `__file__`-based absolute paths and create all required directories upfront
3. **Implement context managers** for PDF documents and add `encoding='utf-8'` to text file operations
