# Pampellone Chatbot Notes



##### Important links:

\- https://softcery.com/lab/building-ai-that-understands-legal-documents

\- https://whisperit.ai/blog/ai-legal-research-rag-techniques-for-lawyers

\- https://www.tencentcloud.com/techpedia/119495





**Pamp\_chatbot Access Token:**

hf\_YSLolvQsYHSQqYQzuRrmahVzekvXDqjWVm





**GROQ API** = gsk\_ffwrq2NLqoh0h5DZ1Pq9WGdyb3FYGEWqfauGMIfxQus1kR2kGjkV







###### **Preparing the law document**



\- The nature of the document means more precision needed in questioning to allow for retrieval

\- Bylaws are not QA-friendly documents. They are retrieval-first documents.



Chunking:

Breaking into clause-aware chunks is the most sensible way to increase retrieval accuracy for by-laws doc

Chunking needed to be different for the first page compared to the main by-laws since the labeling and structure of the first page (definitions) was different and affecting splitting

Two separate regex patterns used to identify the definitions then the bylaws in the doc. Each chunk was labeled accordingly and then appended in order to one main list labeled all\_chunks. This is what will be used.

  Tests of Chunks:

Definition chunks- boundaries

   - Checked for duplicates in clause-id numbers

   - Checked for empty text sections

 



Creating Embeddings and Building Embeddings DB:

Notes: This needs be checked and tested with query to verify accuracy of retrieval with specified model

       \*\*Care needs to be taken that db is not being rebuilt each time. Unless new chunking frameworks are\*\*

       \\\*\\\*made.\\\*\\\*





In building Vectorstore db, text and metadata specified into variables since they were combined in chunks

Query / retrieval tests with score to check if relevant chunks are returned

NOTE: scores not so useful for by-laws doc because there are repeated lines/phrases for clarity other than main definition sections. Results show scores that are similar.



   Testing Retrieval:

   - Provided semantically similar questions to determine if retrieval is consistent or if just working on

     key word matching.

   - Repeated clause titles and phrasing ("quorum") revealed that although embeddings are robust (other

     tests passed) disambiguation required by doing metadata-aware reranking.

   - Disambiguation - creating more depth by labeling section number and name in metadata (e.g.

     distinguishes between 'quorum' sections) (FAILED)

     Note: Deeper levels did not change results, problem lies within the original doc with repeated phrases

     and concepts



###### **Building the Reranker:**

**Notes:**

* *Needed after retrieval from db because repetition in by-laws doc causes ambiguity and close scoring*
* *LLM in reranker can be prompted to disambiguate*



**Reranker model** :

\- Mistral-7B-Instruct (quantized):

Strengths:

* Extremely good at following short, explicit instructions
* Very stable at temperature ≈ 0
* Performs well on classification-style prompts
* Excellent llama.cpp support

Weaknesses:

* Less “world knowledge” than LLaMA-3
* Slightly brittle if prompts are verbose or vague





* \- Prompt extremely important to get the model to behave as desired due to weaknesses
* \- Install LLM (pip install torch transformers accelerate)
* \- Reranking quality is about rubric and prompt design not model intelligence. Paid API not worth it as this point. (How well is 1-5 scores explained and distinguished)
* Model seemed incapable of handling scoring so switched to binary classifier (YES / NO questions)

     Used answerability of retrieved answers ---> then ranking if necessary

* NOTE: *llm  misbehavior is typically from generation controls not reasoning (check parameters)*
* *NOTE: treat llm output as messy  or unreliable use your programming to normalize*



* **Created a pipeline wrapper for retrieval, filtering, reranking/ choosing**
* **Added a score threshold for additional confidence in clause retrieval**

  **NOTE:** Scoring



* ###### **LLM generator causing severe latency- strategies to reduce (Major work here)!!**

  **- Strategies to reduce latency:**

  **- just return clause no generated sentence**

  **- harden prompt rules**



  **Architecture**

  User Query

     ↓

  Vector Search (CPU)

     ↓

  Binary LLM Gate (CPU)

     ↓

  Tie-breaker if needed (CPU)

     ↓

  Decision:

     ├─ Return clause verbatim (0ms)

     ├─ Use cached answer (0ms)

     ├─ GPU generator

     └─ CPU fallback generator (degraded mode)



    \_ Latency strategy employed: used GPU for answer generator only and left CPU for reranker and binary classifier. UPDATE: full pipeline run of one query still super slow



* **Changed CPU model to smaller model (Phi):**

  **-** error with output slicing, so need to rewrite function codes using the model

       - Changed binary classifier to logit based classification to avoid errors with output formatting and

         text since changing model: NO string, text parsing or random output errors

       - **UPDATE:** smaller model was inaccurate regardless code for output either logit or text parsing

       - Tested other small gen models Phi-3.5, this was accurate and smaller



  **UPDATE: Changed CPU model to cross encoder model ( industry standard for gating and ranking)**

  **UPdate: Used model from GROQ** which prevents having to host model on GPU myself. My machine incapable of handling that type of load. 











  #### **TO-DO:  !!!!!**

* update requirements
* update Git



* FAST API
* 







  ##### **NEXT UP:**

* 
* Upload to Git
* U
* 











  Set up for next phase of build:

  Next steps, in order:





  2 Prompt-controlled answer synthesis (citation-first)



  3 LLM selection \& loading (Phi-3 or alternative)



  4 Answer grounding \& hallucination suppression



  5 Deployment hardening

















  #### **DAILY REVIEW:**

* Model generation code
* 





##### 







  ##### **NOTES OF THINGS TO UNDERSTAND:**



* Why this specific embeddings model?
* Distances in embeddings (cosine , L2)
* Why Chroma and not FAISS? Chroma easier to implement for beginner, can incorporate metadatas in a straight forward way. FAISS better for denser more complex documents, faster GPU processing.
* 
* 





  #### **Improvements:**

* Better retrieval methods other than similarity\_search
* Hybrid keyword and similarity search. (Read article book marked in google under RAG Chatbots folder)
* Mongo DB RAG build
* Another way to host for free?







  #### **Additional Features (maybe):**

* Add page number to metadata
* "References to the act should be sought outside of chatbot" - respond with error message or guide to

      Companies ACT



* Enable token streaming - reduces perceived latency
* Provide all relevant clauses to user or all retrieved clauses in order or relevance
* Feedback loop- allow user to rate responses to train model??







  ### **GitHub:**



  “I designed a constrained system intentionally, not accidentally.”



  “This demo is hosted on free infrastructure with limited concurrency.

  Under load, the system prioritizes extractive answers over generative ones.”



  “Designed for acceleration, deployed within free constraints.”



  In your **README**, you should say this out loud:



  CPU models handle classification and routing



  GPU model is reserved for user-facing generation



  Under load, the system prioritizes correctness over fluency



  Extractive answers are returned when appropriate



  This turns a constraint into a design choice.







  \# First time only

  pip install -r requirements.txt

  python build\_db.py



  \# Every time after

  python main.py

