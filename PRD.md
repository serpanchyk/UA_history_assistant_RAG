# Product Requirements Document (PRD)

## **Project Name:** Ukrainian History Student Assistant (Multimodal RAG)

## Problem Statement & Users

### **Users**
Middle and high school students (grades 7–11), university applicants, first-year university students, and history teachers who study or teach the history of Ukraine.

### **Problem We Are Solving**
When preparing for lessons, state exams (DPA/ZNO), self-study, or revision, students often need to quickly find explanations of historical events, figures, or processes, as well as view relevant illustrations (maps, portraits, diagrams, photographs of artifacts) contained in textbooks.

Paper textbooks or PDF versions are inconvenient for fast search: users must manually flip through pages and look for the required map or image.

Our assistant allows users to ask questions in natural Ukrainian and receive a well-grounded textual answer together with relevant illustrations directly sourced from official school textbooks.


## MVP Scope (First Version)

### **What the Application Will Do**
- Accept text-based questions in Ukrainian.
- Retrieve relevant text passages and images from a corpus of Ukrainian school history textbooks.
- Generate answers based **exclusively** on retrieved sources, with citations.
- Display the answer along with text excerpts and thumbnails of relevant images (with zoom capability).

### **Explicitly Out of Scope for MVP**
- Agents, autonomous planning, or tool usage (web search, calculators, etc.).
- Real-time access to external APIs during queries (except for the LLM call).
- Image generation, image editing, or user file uploads.
- Multi-user mode, authentication, or chat history storage.


## Content & Data

### **Data Sources**
Official electronic versions (PDF) of Ukrainian and World History school textbooks for grades 7–11, freely available on the Institute for Educational Content Modernization portal (lib.imzo.gov.ua).

### **Target Volume**
- 250 pages in average in one book.
- 11 textbooks (approximately 2,750 pages total).
- 300 words on one page (420 tokens)
- 1 embedding can take 1024 tokens
- 2 pages of text per chunk
- After processing:
  - 5000 text chunks
  - 2000 images (maps, portraits, diagrams, photographs)

### **Text–Image Linking **

- Captions are identified using **spatial coordinates** of images and text blocks.  
- Text blocks located near or directly below an image are considered its caption.  
- Maintains accurate image–caption pairing even with multiple images per page.


## Example Queries (in Ukrainian)

1. Як вигладала монета викорбована на честь скіфського царя та яким було його правління?
2. Де проходив шлях із варяги у греки?
3. Які причини проголошення четвертого універсалу?
4. Яка схема показує адміністративний поділ УНР?
5. Надай хронологічний перебіг подій Руїни.
6. Які твори мистецтва відображають український романтизм?
7. Що стало наслідком Чигиринських походів турецько-татарської армії?
8. Які завдання перед собою ставила Українська Гельсінська Група?
9. Покажи карту чортківської офензиви?
10. Чия це цитата “*…взяли мене в підозрі за то, що я руські… книги писав, і нещасного 
дня 27 квітня 1849 року арештували мене роз’ярені мадяри, і спочатку 
в міському домі Пряшева заперли…”?*


## Success Criteria

## Evaluation Methodology & Success Criteria

### Dataset Creation (Synthetic Pipeline)
To ensure objective and scalable evaluation, we utilized a **Synthetic Data Generation** pipeline using an auxiliary LLM (`gpt-4o-mini`). This approach minimizes human bias and ensures perfect mapping between queries and "Ground Truth" sources.

**Source Data:**
* **Text Corpus:** Chunked history textbooks (max 1500 chars, 200 overlap).
* **Visual Corpus:** Image captions combined with embedding descriptions from textbooks.

**Query Generation Strategy:**
Generate balanced dataset categorized by modality and complexity to rigorously test the system:

1.  **Text Queries (Semantic & Paraphrased):**
    * *Method:* The LLM was prompted to act as an examiner (NMT style) and generate questions based on specific chunks.
    * *Constraint:* **Strict paraphrasing** was enforced to prevent "keyword leaking" (e.g., transforming "signed the treaty" → "concluded the agreement"), forcing the system to rely on semantic understanding.
    * *Example:* "Які соціальні наслідки мала реформа 1848 року?" (Ground Truth: Chunk #3392).

2.  **Image Queries (Contextual):**
    * *Method:* The LLM generated queries based on image captions but was explicitly **forbidden** from using direct visual descriptors (e.g., "photo", "picture", "show").
    * *Focus:* Queries target the **historical context**, causes, or figures depicted. This forces the model to bridge the gap between abstract concepts and visual data using vector similarity (CLIP/Dense) rather than simple metadata matching.
    * *Example:* "Як виглядав лідер Центральної Ради?" (Ground Truth: `hrushevsky_portrait.jpg`).

---

### 6.2 Retrieval Metrics (Search Quality)
Use **Ablation Study**, to compare **Dense**, **Sparse**, and **Hybrid** search strategies to validate the architectural choices.

* **Hit Rate @ k=5:**
    * *Definition:* The probability that the specific "Ground Truth" chunk or image appears in the top-5 retrieved results.
    * *Success Target:* **> 80%** for Hybrid Search.
* **MRR (Mean Reciprocal Rank):**
    * *Definition:* Evaluates the ranking quality. If the correct document is at position 1, the score is 1.0; at position 2, the score is 0.5.
    * *Success Target:* **> 0.65** (implies the correct answer is consistently in the top 1-2 results).

---

### Generation Metrics (RAG Quality)
To assess the final answer generated by the primary LLM (`gpt-4o`), we employ the **LLM-as-a-Judge** pattern alongside semantic analysis:

1.  **Faithfulness (Groundedness):**
    * *Method:* An LLM Judge verifies if the generated answer is derived *solely* from the retrieved context without hallucinations.
    * *Target:* **> 0.90**.
2.  **Citation Correctness:**
    * *Method:* An LLM Judge verifies that every claim is explicitly attributed to a source, and that the cited source actually contains that information.
    * *Target:* **> 0.85**.
3.  **Answer Relevance:**
    * *Method:* An LLM Judge determines if the response actually addresses the specific user question (ignoring answers that are factually correct but irrelevant to the query).
    * *Target:* **> 0.85**.
4.  **Semantic Similarity:**
    * *Method:* Calculates the cosine similarity between the vector embedding of the Generated Answer and the Ground Truth source text.


## Interface Expectations

- Text input field + **Submit** button.
- Answer panel (text + citations).
- Sources panel (text excerpts with textbook title and page number).
- Images panel (thumbnails with captions, click to enlarge).


## Initial Technical Choices

- **LLM:** gpt-4o and gpt-5 for aux
- **Embeddings:**
  - Text: multilingual model — 'BAAI/bge-m3' (keyword embeddings could be great addition for getting better matches)
  - Images: OpenAI CLIP
- **Vector Store:** QDRANT (allows to save embeddings with texts)
- **Framework:** LangChain (indexing, retrieval, generation)
- **UI:** Gradio

