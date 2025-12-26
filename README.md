# Product Requirements Document (PRD)

## **Project Name:** Ukrainian History Student Assistant (Multimodal RAG)

## Problem Statement & Users

### **Users**
Middle and high school students (grades 7–11), university applicants, first-year university students, and history teachers who study or teach the history of Ukraine.

### **Problem We Are Solving**
When preparing for lessons, state exams (DPA/ZNO), self-study, or revision, students often need to quickly find explanations of historical events, figures, or processes, as well as view relevant illustrations (maps, portraits, diagrams, photographs of artifacts) contained in textbooks.

Paper textbooks or PDF versions are inconvenient for fast search: users must manually flip through pages and look for the required map or image.

Our assistant allows users to ask questions in natural Ukrainian and receive a well-grounded textual answer together with relevant illustrations directly sourced from official school textbooks.

---

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

---

## Content & Data

### **Data Sources**
Official electronic versions (PDF) of Ukrainian and World History school textbooks for grades 7–11, freely available on the Institute for Educational Content Modernization portal (lib.imzo.gov.ua).

### **Target Volume**
- 10–15 textbooks (approximately 3,000–4,500 pages total).
- After processing:
  - 150–225 text chunks
  - 2000–3000 images (maps, portraits, diagrams, photographs)

### **Text–Image Linking**
Images are linked to text via page numbers, automatically extracted titles/captions (e.g., “Fig. 1. Map of Kyivan Rus”, “Photo. Taras Shevchenko”), and PDF metadata.

---

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

---

## Success Criteria

- **Search Quality:** For 70%+ of test queries, the correct source (text or image) appears in the top-5 results.
- **Answer Faithfulness:** Answers contain only information from the retrieved sources and include citations. If data is missing, the system responds: *“Insufficient information.”*
- **UI Usability:** Users can view the answer, citations, and enlarged images with a single click.

---

## Interface Expectations

- Text input field + **Submit** button.
- Answer panel (text + citations).
- Sources panel (text excerpts with textbook title and page number).
- Images panel (thumbnails with captions, click to enlarge).

---

## Initial Technical Choices

- **LLM:** `gemma-3-12b-pt` (3-shot) (3rd place in ukrainian llm leaderboard)
- **Embeddings:**
  - Text: multilingual model — 'BAAI/bge-m3' (keyword embeddings could be great addition for getting better matches)
  - Images: OpenAI CLIP
- **Vector Store:** FAISS (local, efficient for fast embedding search, supports multimodal indexing)
- **Framework:** LangChain (indexing, retrieval, generation)
- **UI:** Gradio
