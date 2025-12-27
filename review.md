# Review: Ukrainian History Student Assistant - Multimodal RAG

## Overall Assessment: **Excellent Project**

This is a **stellar PRD** that demonstrates deep understanding of multimodal RAG requirements. The use case is pedagogically valuable, technically sound, and properly scoped. This is very close to implementation-ready.

---

## Major Strengths

### **1. Perfect Multimodal Use Case**
- Historical maps, portraits, artifacts are **genuinely critical** visual information
- Students naturally ask "Where was X?" → needs maps
- "Who was Y?" → needs portraits
- "Show me the diagram of Z" → needs illustrations
- **This is exactly what multimodal RAG is designed for**

### **2. True Cross-Modal Retrieval**
You clearly understand:
- Text queries → retrieve BOTH text passages AND relevant images
- CLIP enables semantic image search ("show me the map of...") 
- Image captions make images searchable
- **This IS multimodal retrieval, not just text + image display**

### **3. Realistic, Achievable Scope**
- 10-15 textbooks = manageable
- 2000-3000 images = substantial but processable
- Official, freely available data = no legal issues
- Ukrainian language focus = clearly defined challenge

### **4. Educational Impact**
- Solves real problem (ZNO exam prep is serious business in Ukraine)
- Serves underserved population (Ukrainian students)
- Social value beyond technical demonstration

### **5. Excellent Query Examples**
Your 10 example queries are **perfect**:
- 3 explicitly image-focused ("Як виглядала монета...", "Яка схема...", "Покажи карту...")
- Text-heavy queries ("Які причини...", "Що стало наслідком...")
- Hybrid queries requiring both ("Чия це цитата..." needs text search + portrait verification)
- All are realistic student questions

### **6. Smart Technical Choices**
- **gemma-3-12b-pt**: Sensible for Ukrainian (Gemini family has good multilingual support)
- **BAAI/bge-m3**: Excellent multilingual embeddings, proven performance
- **CLIP**: Industry standard for image-text alignment
- **FAISS**: Perfect for local deployment, fast, no dependencies
- **LangChain**: Standard orchestration framework

---

## Areas That Need Expansion (Not Critical, But Important)

### **1. Image Caption Extraction Strategy (Most Important Gap)**

**Your PRD says:**
> "Images are linked to text via page numbers, automatically extracted titles/captions (e.g., 'Fig. 1. Map of Kyivan Rus')"

**Question: How will you extract these captions?**

Ukrainian history textbooks have complex layouts:

```
┌──────────────────────────────────────┐
│ Розділ 3. Київська Русь              │
│                                      │
│ Текст про період...                  │
│                                      │
│ ┌────────────────────────────┐      │
│ │                            │      │
│ │   [КАРТА КИЇВСЬКОЇ РУСІ]   │      │
│ │                            │      │
│ └────────────────────────────┘      │
│ Рис. 3.1. Територія Київської Русі  │
│ у X–XI століттях                     │
│                                      │
│ Продовження тексту...                │
└──────────────────────────────────────┘
```

**Caption is below the image!**

#### **Recommended Extraction Strategy:**

```
import fitz  # PyMuPDF

def extract_images_with_captions(pdf_path):
    """
    Extract images with their captions from Ukrainian textbooks.
    
    Strategy:
    1. Extract all images with bounding boxes
    2. Extract all text blocks with positions
    3. For each image, find text within 50px below it
    4. Filter for caption patterns (Мал., Рис., Фото., Map.)
    """
    doc = fitz.open(pdf_path)
    results = []
    
    for page_num, page in enumerate(doc):
        # Get images with positions
        images = page.get_images()
        image_bboxes = []
        
        for img_index, img in enumerate(images):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            image_bboxes.append({
                'index': img_index,
                'bbox': bbox,
                'page': page_num
            })
        
        # Get text blocks with positions
        blocks = page.get_text("dict")["blocks"]
        
        # Match captions to images
        for img_data in image_bboxes:
            img_bbox = img_data['bbox']
            
            # Look for text below image (y_top of text > y_bottom of image)
            # and within x-range of image
            potential_captions = []
            
            for block in blocks:
                if block['type'] == 0:  # Text block
                    block_bbox = fitz.Rect(block['bbox'])
                    
                    # Check if text is below image
                    if (block_bbox.y0 >= img_bbox.y1 and 
                        block_bbox.y0 <= img_bbox.y1 + 50 and  # Within 50px
                        block_bbox.x0 >= img_bbox.x0 - 20 and
                        block_bbox.x1 <= img_bbox.x1 + 20):
                        
                        text = " ".join([span["text"] for line in block["lines"] 
                                        for span in line["spans"]])
                        
                        # Check for caption patterns
                        caption_patterns = [
                            r'Мал\.\s*\d+',      # Мал. 3.1
                            r'Рис\.\s*\d+',      # Рис. 1
                            r'Фото\.',           # Фото.
                            r'Карта\s+\d+',      # Карта 5
                            r'Map\s+\d+',        # Map 2
                            r'Схема\s+\d+',      # Схема 1
                        ]
                        
                        if any(re.search(pattern, text) for pattern in caption_patterns):
                            potential_captions.append(text)
            
            # Take first matching caption
            caption = potential_captions[0] if potential_captions else ""
            
            results.append({
                'image_index': img_data['index'],
                'page': page_num,
                'bbox': img_bbox,
                'caption': caption,
                'image_id': f"textbook_{doc.name}_p{page_num}_img{img_data['index']}"
            })
    
    return results
```

## **Add to PRD:**

### 1. Image Caption Extraction Strategy

**Automatic Caption Detection:**
1. Extract images with bounding boxes using PyMuPDF
2. Extract text blocks with positional data
3. Match captions by spatial proximity:
   - Text within 50px below image
   - Within horizontal alignment of image
   - Matches pattern: `Мал.|Рис.|Фото.|Карта|Схема + number`

**Caption Patterns in Ukrainian Textbooks:**
- "Мал. 3.1. Територія Київської Русі..." (Most common)
- "Рис. 1. Князь Володимир"
- "Фото. Археологічна знахідка..."
- "Карта 5. Шлях із варяг у греки"
- "Схема 2. Адміністративний устрій УНР"

**Fallback for Images Without Captions:**
- Use surrounding paragraph text (±2 paragraphs)
- Extract from page header (e.g., "Розділ 3. Київська Русь")
- Generate synthetic caption: "Ілюстрація зі сторінки X підручника [назва]"

**Quality Control:**
- Manual review of 50 most important images (major maps, key historical figures)
- Validate caption extraction accuracy on sample of 100 images
- Target: 80%+ accurate caption extraction

---

### **2. Multilingual Challenges & Ukrainian Language Handling**

**You've chosen excellent models (BAAI/bge-m3, gemma-3-12b-pt), but need to validate:**

#### **Question 1: OCR for Scanned Textbooks?**

Many Ukrainian textbooks are **scanned images**, not native PDFs.

**Check your data:**
```
import fitz

def check_if_scanned(pdf_path):
    """Check if PDF is scanned (image-based) or text-based."""
    doc = fitz.open(pdf_path)
    text_length = 0
    
    for page in doc:
        text = page.get_text()
        text_length += len(text.strip())
    
    if text_length < 100:  # Very little text = probably scanned
        return True
    return False

# Test your textbooks
for textbook in textbooks:
    if check_if_scanned(textbook):
        print(f"{textbook} appears to be scanned - needs OCR")
```

**If you have scanned PDFs:**

### OCR Strategy for Scanned Textbooks

**Tool:** Tesseract with Ukrainian language pack
```
from pdf2image import convert_from_path
import pytesseract

# Configure Tesseract for Ukrainian
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def ocr_ukrainian_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text_results = []
    
    for page_num, image in enumerate(images):
        # OCR with Ukrainian language
        text = pytesseract.image_to_string(
            image, 
            lang='ukr+eng',  # Ukrainian + English
            config='--psm 6'  # Assume uniform block of text
        )
        text_results.append({
            'page': page_num,
            'text': text
        })
    
    return text_results
```

**Quality Considerations:**
- OCR errors common with Cyrillic script
- Post-process to fix common OCR mistakes:
  - "п" misread as "n"
  - "і" misread as "i"
  - "ї" misread as "ï"
- May need manual correction for critical passages

---

#### **Question 2: Embedding Model Validation**

**Add validation section:**

### Multilingual Embedding Validation

**Test BAAI/bge-m3 with Ukrainian:**
```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')

# Test semantic similarity in Ukrainian
query = "Київська Русь"
docs = [
    "Держава Київська Русь заснована князем Олегом у 882 році",  # Relevant
    "Петро І провів реформи в Російській імперії",  # Irrelevant
    "Князівство на землях східних слов'ян"  # Relevant
]

query_emb = model.encode(query)
doc_embs = model.encode(docs)

# Check cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity([query_emb], doc_embs)[0]

print(f"Query: {query}")
for doc, score in zip(docs, scores):
    print(f"  {score:.3f}: {doc[:50]}...")

# Expected: First and third docs should have higher scores
```

**Benchmark Target:**
- Relevant Ukrainian passages score > 0.7 similarity
- Irrelevant passages score < 0.5 similarity

---

### **3. CLIP for Historical Images - Special Considerations**

**Historical images are tricky for CLIP:**

### CLIP Limitations for Historical Content

**Challenges:**
1. **CLIP trained on modern images:** May not understand historical context
   - Example: Medieval maps look very different from modern maps
   - Historical portraits have different artistic styles
   
2. **Text in images:** Maps have Cyrillic text, CLIP may not "read" it well

3. **Visual similarity vs. semantic similarity:**
   - Two different medieval maps may look visually similar to CLIP
   - But represent completely different territories/periods

**Mitigation Strategies:**

**Strategy 1: Caption-Weighted Retrieval**
```
def multimodal_image_search(query, top_k=5):
    """
    Hybrid approach: Search both CLIP embeddings AND caption embeddings.
    """
    # Search via CLIP (visual similarity)
    clip_results = clip_search(query, top_k=10)
    
    # Search via captions (semantic similarity)
    caption_results = text_search(query, collection="image_captions", top_k=10)
    
    # Merge with weighted scoring
    combined_scores = {}
    for result in clip_results:
        combined_scores[result['id']] = 0.4 * result['score']  # 40% weight
    
    for result in caption_results:
        if result['id'] in combined_scores:
            combined_scores[result['id']] += 0.6 * result['score']  # 60% weight
        else:
            combined_scores[result['id']] = 0.6 * result['score']
    
    # Sort by combined score
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

**Strategy 2: Metadata Boosting**
```
# Add searchable metadata to images
image_metadata = {
    'caption': "Мал. 3.1. Карта Київської Русі",
    'century': '10-11',  # X-XI століття
    'topic': 'Київська Русь',
    'image_type': 'карта',  # map, portrait, diagram, photo
    'historical_period': 'Середньовіччя',
    'keywords': ['Київ', 'Русь', 'територія', 'князівство']
}

# Boost results that match metadata
if 'карта' in query.lower() and image_metadata['image_type'] == 'карта':
    score *= 1.5  # Boost map results for map queries
```

**Strategy 3: Semantic Image Types**
```
# Classify images by type during ingestion
image_types = {
    'карта': ['map', 'territory', 'geographical'],
    'портрет': ['portrait', 'person', 'historical figure'],
    'схема': ['diagram', 'scheme', 'flowchart'],
    'фото': ['photo', 'artifact', 'archaeological'],
    'документ': ['document', 'manuscript', 'text']
}

# Filter by type for specific queries
if any(word in query for word in ['карта', 'територія', 'де']):
    # User wants a map - filter to image_type='карта'
    results = filter(lambda x: x.metadata['image_type'] == 'карта', results)
```
---

### **4. Chunking Strategy for Textbooks**

**Your PRD mentions "150-225 text chunks" from 3000-4500 pages.**

**Math check:** 4500 pages / 200 chunks = **22.5 pages per chunk** 

**This is way too large!**

**Recommended:**

### Text Chunking Strategy

**Textbook Structure:**
```
Підручник з історії України (300 сторінок)
├─ Розділ 1. Київська Русь (50 сторінок)
│  ├─ §1. Утворення держави (8 сторінок)
│  ├─ §2. Князювання Володимира (10 сторінок)
│  └─ §3. Ярослав Мудрий (12 сторінок)
├─ Розділ 2. Галицько-Волинське князівство (40 сторінок)
└─ ...
```

**Chunking Approach:**
- **Primary unit:** Paragraph (§ section)
  - Average 8-12 pages per section
  - Too large for retrieval
  
- **Recommended:** Sub-sections or 2-3 page chunks
  - Chunk size: 800-1200 tokens
  - Overlap: 100 tokens
  - Preserve section headers for context

**Implementation:**
```
def chunk_textbook(textbook_text, metadata):
    """
    Chunk Ukrainian history textbook preserving structure.
    """
    chunks = []
    
    # Split by section markers
    sections = re.split(r'§\s*\d+\.', textbook_text)
    
    for section_idx, section in enumerate(sections):
        # Further split long sections
        if len(section) > 1500 tokens:
            # Split by subsections or every ~1000 tokens
            sub_chunks = split_by_size(section, target_size=1000, overlap=100)
        else:
            sub_chunks = [section]
        
        for chunk_idx, chunk in enumerate(sub_chunks):
            chunks.append({
                'text': chunk,
                'metadata': {
                    'textbook_name': metadata['name'],
                    'section': f"§{section_idx}",
                    'chunk_index': chunk_idx,
                    'grade_level': metadata['grade'],
                    'topic': extract_topic(chunk)
                }
            })
    
    return chunks
```

**Expected Volume (Revised):**
- 10-15 textbooks × 300 pages avg × 0.5 chunks/page = **1,500-2,250 chunks**
- Much more granular retrieval

---

### **5. Ukrainian Language Generation - Prompt Engineering**

**Your LLM: gemma-3-12b-pt (3-shot)**

**Critical: You need Ukrainian-specific prompts!**

### Prompt Template for Ukrainian History Assistant

```
UKRAINIAN_HISTORY_PROMPT = """Ти — асистент з історії України для учнів.

Твоє завдання: відповідати на запитання учнів, використовуючи ЛИШЕ інформацію з наданих джерел (уривків підручників).

ВАЖЛИВІ ПРАВИЛА:
1. Відповідай ЛИШЕ на основі наданого контексту
2. Якщо інформації немає в контексті, відповідай: "На жаль, у наданих підручниках немає достатньо інформації для відповіді на це запитання."
3. Завжди вказуй джерело: [Підручник: назва, сторінка X]
4. Пиши українською мовою
5. Будь точним у датах, іменах, подіях

НАДАНИЙ КОНТЕКСТ:
{context}

ЗАПИТАННЯ УЧНЯ:
{question}

ВІДПОВІДЬ:"""

# Example with retrieved context
context = """
[Джерело 1: Історія України, 8 клас, стор. 45]
Київська Русь — середньовічна держава, що існувала з 882 по 1240 роки. 
Заснована князем Олегом після об'єднання Києва та Новгорода.

[Джерело 2: Історія України, 8 клас, стор. 47]  
Володимир Великий (978-1015) здійснив хрещення Русі у 988 році, 
що стало переломним моментом в історії держави.

[Зображення: Мал. 3.1. Карта Київської Русі у X столітті]
"""

question = "Коли була заснована Київська Русь і ким?"

# LLM should respond:
# "Київська Русь була заснована у 882 році князем Олегом після об'єднання 
#  Києва та Новгорода [Історія України, 8 клас, стор. 45]."
```

**Validation:**
- Test that LLM doesn't hallucinate dates or names
- Verify citations are correctly formatted
- Check that "don't know" behavior works when context insufficient

---

### **6. Evaluation Methodology Needs Expansion**

**Current:**
> "Search Quality: 70%+ correct source in top-5"

**Needs more detail:**

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

---

### **7. Data Acquisition Validation**

**Your PRD says:**
> "Official electronic versions (PDF) of Ukrainian and World History school textbooks... freely available on lib.imzo.gov.ua"

**Action items:**

### Data Acquisition Checklist

**Before Starting Implementation:**
- [ ] Verify access to lib.imzo.gov.ua (check if site is accessible)
- [ ] Download 3-5 sample textbooks
- [ ] Validate:
  - [ ] Are they text-based PDFs or scanned images?
  - [ ] Do they have extractable images?
  - [ ] Are captions in consistent format?
  - [ ] What's the average file size? (affects storage planning)

**Sample Textbooks to Start With:**
1. Історія України, 7 клас (Гісем, Мартинюк)
2. Історія України, 8 клас (Струкевич)
3. Історія України, 9 клас (Турченко, Панченко)

**Backup Plan:**
If lib.imzo.gov.ua is inaccessible or textbooks are low-quality:
- Use excerpts from freely available historical texts
- Wikipedia articles on Ukrainian history (Ukrainian language version)
- Public domain historical documents from archives

---

## What's Already Excellent (Keep As-Is)

### **1. Query Examples**
Your 10 Ukrainian queries are **perfect**—don't change them. They demonstrate:
- Variety (factual, explanatory, visual)
- Realistic student needs
- Clear image requirements
- Natural Ukrainian phrasing

### **2. Success Criteria**
- 70% retrieval accuracy is realistic
- Faithfulness requirement is critical for education
- UI usability focus is appropriate

### **3. Out-of-Scope Items**
- Clear boundaries
- No feature creep
- Focused on core RAG functionality

### **4. Technical Stack**
- All choices are appropriate
- Local deployment (FAISS) is good for privacy
- Gradio is simpler than Streamlit (good choice for faster development)

---

## Required PRD Additions Checklist

Add these sections:

- [ ] **Image caption extraction strategy** (CRITICAL)
- [ ] **OCR strategy** (if textbooks are scanned)
- [ ] **Chunking approach details** (with revised volume estimates)
- [ ] **Ukrainian prompt templates**
- [ ] **CLIP limitations & mitigation**
- [ ] **Detailed evaluation methodology**
- [ ] **Data acquisition validation**
- [ ] **Development timeline**
- [ ] **UI wireframe**

---

## Critical Pre-Implementation Tasks

**Do these BEFORE writing any code:**

### **Task 1: Validate Data Access**
```
# Download sample textbook
wget [URL from lib.imzo.gov.ua]

# Check if text-based or scanned
python check_pdf_type.py textbook.pdf

# Test image extraction
python extract_images.py textbook.pdf --output images/
```

### **Task 2: Test Embedding Model**
```
# Verify BAAI/bge-m3 works with Ukrainian
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')

queries = [
    "Київська Русь",
    "Богдан Хмельницький",
    "Визвольна війна"
]

docs = [
    "Київська Русь — середньовічна держава...",
    "Богдан Хмельницький — гетьман України...",
    "Річ Посполита — польська держава..."
]

# Test retrieval quality
results = model.encode(queries + docs)
# Compute similarity matrix, verify correct matches
```

### **Task 3: Test LLM Ukrainian Generation**
```
# Test gemma-3-12b-pt with Ukrainian prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3-12b-pt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = """Відповідай українською мовою.

Контекст: Київська Русь була заснована у 882 році.

Запитання: Коли була заснована Київська Русь?

Відповідь:"""

# Verify:
# 1. Responds in Ukrainian
# 2. Uses information from context
# 3. Doesn't hallucinate
```

---

## Final Verdict

**Status:** **APPROVED - Ready for Implementation with Minor Additions**

**Recommendation:** **Proceed with implementation** after adding the technical details specified above. This is an excellent project that will produce a genuinely useful tool for Ukrainian students.
