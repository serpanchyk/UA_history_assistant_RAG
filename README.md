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
  - 1375 text chunks
  - 3000–5000 images (maps, portraits, diagrams, photographs)

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

### Evaluation Methodology (Detailed)

**Evaluation Dataset Creation:**

**Phase 1: Query Collection (40 queries)**
- Collect from: Past ZNO exam questions (historical knowledge section)
- Teacher-provided common student questions
- Synthetic queries covering different periods/topics

**Query Distribution:**
- 10 queries: Factual (dates, names, definitions)
  - "Коли відбулося хрещення Русі?"
  - "Хто був гетьманом під час Визвольної війни?"
  
- 10 queries: Explanatory (causes, consequences)
  - "Які причини падіння Київської Русі?"
  - "Що стало наслідком Переяславської ради?"
  
- 10 queries: Image-focused (maps, portraits, diagrams)
  - "Покажи карту козацьких земель у XVII столітті"
  - "Як виглядав гетьман Богдан Хмельницький?"
  
- 10 queries: Hybrid (text + image verification)
  - "Яка схема показує адміністративний устрій Гетьманщини?"
  - "Хто автор цитати [quote] і покажи його портрет"

**Phase 2: Gold Standard Annotation**
For each query, manually label:
- Expected text chunks (by textbook + page number)
- Expected images (by figure number)
- Correct answer (reference answer from textbook)

**Metrics:**

**Retrieval Metrics:**
1. **Text Retrieval:**
   - Recall@5: Is correct text passage in top-5?
   - MRR (Mean Reciprocal Rank): Position of first relevant result
   - Target: Recall@5 > 75%, MRR > 0.6

2. **Image Retrieval:**
   - Image Hit Rate@3: Is correct image in top-3?
   - Image Precision: Are top-3 images relevant?
   - Target: Hit Rate > 70%

3. **Hybrid Queries:**
   - Both-Retrieved Rate: Both text AND image in top-5
   - Target: > 65%

**Generation Metrics:**
1. **Faithfulness (Critical for Education):**
   - Manual review: Does answer contain ONLY info from sources?
   - Binary scoring: Faithful (1) or Hallucinated (0)
   - Target: 100% faithful (zero hallucinations acceptable for student tool)

2. **Citation Accuracy:**
   - Are all citations correct (right textbook + page)?
   - Target: 95%+

3. **Completeness:**
   - Does answer address the full question?
   - 3-point scale: Complete (2), Partial (1), Incomplete (0)
   - Target: Average > 1.5

4. **Ukrainian Language Quality:**
   - Grammar correctness
   - Natural phrasing (not machine translation quality)
   - Historical terminology accuracy
   - Manual review by Ukrainian teacher/historian

**Baseline Comparisons:**
- Compare vs. Ctrl+F keyword search in PDFs
- Compare vs. ChatGPT without RAG (to show hallucination problem)
- Compare vs. Google Search (to show grounded citations)

**Error Analysis:**
Track failure modes:
- Incorrect OCR → wrong text retrieval
- Caption mismatch → wrong image retrieval
- LLM hallucination despite correct context
- Missing information in corpus


## Interface Expectations

- Text input field + **Submit** button.
- Answer panel (text + citations).
- Sources panel (text excerpts with textbook title and page number).
- Images panel (thumbnails with captions, click to enlarge).


## Initial Technical Choices

- **LLM:** MamayLM-Gemma-3-4B-IT-v1.0.Q4_K_S (1st place in Ukrainian Language Leaderboard. Optimised version for my pc, 5 tokens/second)
- **Embeddings:**
  - Text: multilingual model — 'BAAI/bge-m3' (keyword embeddings could be great addition for getting better matches)
  - Images: OpenAI CLIP
- **Vector Store:** QDRANT (allows to save embeddings with texts)
- **Framework:** LangChain (indexing, retrieval, generation)
- **UI:** Streamlit
