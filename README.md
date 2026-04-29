# Pampellone Villas By-Laws Assistant

A production-deployed RAG chatbot that answers questions about a real HOA legal document, grounded strictly in source text, with sub-second retrieval, and **built and hosted at zero cost.**

---

## The Goal

This is my first fully deployed AI application. I set myself four constraints from the start:

1. **Useful** — solve a real problem, not a toy one
2. **Trustworthy** — answers must be traceable to source text; the app should refuse to answer rather than guess
3. **Fast** — latency was a first-class concern, not an afterthought
4. **Free** — zero spend, from development through production deployment

Every architectural decision flows from those constraints.

---

## What It Does

- Answers natural-language questions about the Pampellone Villas LTD By-laws
- Cites the specific clause that answered the question
- Lets users expand and read the raw clause text for full transparency
- Refuses to answer if no relevant clause exists (instead of making something up)

---

## How It Works — The RAG Pipeline

The pipeline retrieves only what it needs, applies two relevance filters to avoid false positives, attempts a query rewrite if the filters find nothing, then constrains the LLM to answer strictly from the matched clause:


```
User Question
      │
      ▼
┌─────────────────────────────┐
│  1. VECTOR SEARCH           │  Find the top 3 chunks from the by-laws
│     ChromaDB + MiniLM       │  document most similar to the query
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  2. DISTANCE GATE           │  Reject if no chunk is close enough
│     L2 distance check       │  ("Your question isn't in the by-laws")
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  3. CROSS-ENCODER RERANK    │  Score each chunk against the question
│     ms-marco MiniLM         │  more precisely; drop irrelevant ones
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  4. QUERY REWRITE FALLBACK  │  If cross-encoder finds nothing, rewrite
│     Groq / Llama 3.1 8B     │  the query in legal language and retry
│                             │  steps 1 & 3 once. If still nothing,
│                             │  return a graceful refusal.
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  5. LLM ANSWER GENERATION   │  Pass the best clause + question to
│     Groq / Llama 3.3 70B    │  LLaMA; answer strictly from clause
└─────────────────────────────┘
      │
      ▼
   Answer + Clause ID + Raw Clause Text
```

### Multiple retrieval and filtering stages

The first stage (vector search) is fast but imprecise. It works by comparing mathematical representations of meaning, which can return *related-but-wrong* chunks. The second stage (L2 distance check) acts as a gate to block retrieved clauses with a distance score greater than a calibrated maximum distance (MAX_RETRIEVAL_DIST). The third stage (cross-encoder) re-reads the question and each candidate chunk together and gives a more accurate relevance score. This three-stage approach catches false positives and ensures the final answer is built on genuinely relevant text.

### Query rewrite fallback

The cross-encoder filter can fail when a user phrases a question in casual, everyday language that doesn't match the formal terminology used in the by-laws. For example, a user might ask *"Can I build a fence in my garden?"* while the relevant clause uses the term *"perimeter enclosures."* The embedding model may still retrieve the right chunk in step 1, but the cross-encoder then rejects it because the phrasing mismatch is too large.

When this happens, rather than immediately returning a refusal, the pipeline makes one attempt to recover. It sends the original query to a lightweight LLM (`llama-3.1-8b-instant`) with a short prompt instructing it to rephrase the question using formal legal and property by-law language. The rewritten query is then used to re-run the vector search and cross-encoder. If relevant clauses are found this time, the rewritten query is also passed to the answer generator so the LLM response is aligned with the legal phrasing of the matched clause. If the rewrite still finds nothing, the pipeline falls through to a graceful refusal as normal.

A few design details worth noting:

- **Runs at most once per user query.** The fallback is a pair of plain `if` statements, not a loop. There is no mechanism to retry again.
- **Uses a smaller model for the rewrite.** `llama-3.1-8b-instant` is faster and lighter than the main answer model (`llama-3.3-70b-versatile`), and the rewrite task is simple enough not to need a large model. Token output is also capped aggressively at 60 tokens since rewrites are short.
- **Fails silently.** If the Groq API call for the rewrite fails for any reason (timeout, rate limit, etc.), the function returns the original query and the pipeline proceeds to the normal refusal message. The user never sees an error from the rewrite step itself.

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Vector Database | [ChromaDB](https://www.trychroma.com/) | Free |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) | Free |
| Reranking Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace) | Free |
| LLM (answers) | [Groq API](https://console.groq.com/) — `llama-3.3-70b-versatile` | Free tier |
| LLM (rewrite) | [Groq API](https://console.groq.com/) — `llama-3.1-8b-instant` | Free tier |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) | Free |
| Frontend | Vanilla HTML/CSS/JS | Free |
| Deployment | Google Cloud Run | Free tier |
| Containerisation | Docker | Free |


**Total cost to build and run: $0.**

The two HuggingFace models are downloaded once into the Docker image at build time, eliminating cold-start latency on Cloud Run and removing any runtime dependency on the model hub. The Groq API provides a free tier with generous rate limits and notably fast inference speeds.

---

## Project Structure

```
├── config.py          # All settings in one place (paths, thresholds, model names, other constants)
├── main.py            # FastAPI app — HTTP routes and request/response handling
├── retrieval.py       # The full RAG pipeline (vector search → rerank → rewrite fallback → answer)
├── llm.py             # Groq API integration, prompt construction, and query rewriter
├── vectorstore.py     # Builds and loads the ChromaDB vector database
├── index.html         # The frontend — served directly by FastAPI
├── Dockerfile         # Container definition (includes model pre-download)
└── data/
    └── Pampellone_by_laws.pdf   # The source document
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/)
- The Pampellone Villas by-laws PDF in a `/data` folder

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your environment

Create a `.env` file in the project root:

```
GROQ_API_KEY="your_key_here"
```

### 3. Build the vector database

This only needs to be done once. It reads the PDF, splits it into chunks, and stores them in ChromaDB:

```bash
python vectorstore.py
```

### 4. Run the app

```bash
python main.py
```

Open your browser at `http://localhost:8080`.

---

## Running with Docker

```bash
docker build -t bylaws-chatbot .
docker run -p 8080:8080 --env-file .env bylaws-chatbot
```

The Dockerfile pre-downloads both HuggingFace models during the build step, so the container starts instantly without any model downloads at runtime.

---

## Configuration

All tunable settings live in `config.py`:

| Setting | Default | What it controls |
|---|---|---|
| `RETRIEVAL_K` | `3` | How many chunks to fetch from ChromaDB |
| `MAX_RETRIEVAL_DIST` | `1.2` | Max L2 distance to consider a chunk relevant |
| `RELEVANCE_THRESHOLD` | `2.0` | Min cross-encoder score to pass the filter |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | The LLM used for answer generation |
| `GROQ_REWRITE_MODEL` | `llama-3.1-8b-instant` | The LLM used for query rewriting (fallback only) |
| `GROQ_MAX_TOKENS` | `300` | Max length of the LLM's response |

---

## Design Decisions

**Why two retrieval stages instead of one?**
By-laws documents are replete with repetitions. This may render one stage of retrieval inadequate. Vector search (stage 1) is fast but imprecise since it compares embeddings and can surface *related-but-wrong* chunks. The cross-encoder (stage 3) re-reads the question and each candidate together, catching false positives that vector similarity misses. Running the cross-encoder on only 3 pre-filtered chunks keeps it fast; running it on the full document would be unusably slow.

**Why add a query rewrite fallback?**
The cross-encoder is precise but depends on both the question and clause using similar language. Users naturally ask questions in plain conversational language, while by-laws are written in formal legal prose. When this gap is too large, the cross-encoder correctly rejects the retrieved chunks even if one of them is actually the right answer. The rewrite step bridges that gap by translating the user's phrasing into legal terminology before retrying. It runs only when the cross-encoder has already failed, and only once, so the latency cost is paid only when necessary. A smaller, faster model handles the rewrite to keep that cost low.

**Why bake the models into the Docker image?**
Cold start latency was a real concern on Cloud Run, which can spin down idle containers. Downloading two HuggingFace models at runtime adds 30–60 seconds to the first request. Pre-downloading them at build time (using `RUN python -c "..."` in the Dockerfile) eliminates this entirely and the container can start with everything it needs already on disk.

**Why set `TRANSFORMERS_OFFLINE=1` at runtime?**
After baking the models in, there's no reason for the running container to ever reach the HuggingFace hub. Setting the offline flag makes that guarantee explicit and removes a class of potential runtime failures.

**Why Groq instead of OpenAI?**
Groq's free tier runs `llama-3.3-70b-versatile`, a model that competes with GPT-4 on many benchmarks and its inference speed is significantly faster than OpenAI's API at equivalent quality. For a latency-sensitive, zero-budget project, it was the obvious choice.

**Why refuse to answer instead of guessing?**
Trust is the most important feature of a legal/governance tool. An answer that says "this isn't covered by the by-laws" is far more useful than a confidently wrong one. The two-stage gate (distance threshold + cross-encoder score) means the app says "I don't know" reliably rather than occasionally. The query rewrite fallback adds one recovery attempt before reaching that refusal, without compromising the trustworthiness of the final answer.

**Why not just send the whole document to the LLM?**
Token limits aside, constraining the model to a single retrieved clause is itself a hallucination-reduction technique. The less context a model has to drift from, the less it can invent. The design also took into consideration future scaling; adding updated by-laws or the Companies Act of Trinidad and Tobago. Feeding full documents such as these would be impossible or impractical while still maintaining the designated architectural constraints.

---

## API Reference

**`POST /ask`**

Request:
```json
{ "question": "Who can call a special meeting?" }
```

Response:
```json
{
  "answer": "According to Clause 5.03, a special meeting may be called by...",
  "clause_id": "5.03",
  "clause_text": "Special meetings of the Members may be called..."
}
```

**`GET /health`** — Returns `{"status": "OK"}` if the server is running.

---

## Acknowledgements

Built with [FastAPI](https://fastapi.tiangolo.com/), [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Sentence Transformers](https://www.sbert.net/), and [Groq](https://groq.com/).
