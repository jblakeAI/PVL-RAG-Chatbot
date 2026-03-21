"""
ingestion.py

Handles document ingestion from raw PDF files and converts them into useable chunks. 


Functions: 

load_pdf_text  -  extracts raw text from PDF file as a single concatenated string. 

normalize_text-   collapses excessive blank lines and removes leading white spaces.

normalize_leading_clause_whitespace  - removes leading white space where it exists before clause numbers 
                                        (e.g. '  2.04 Execution' → '2.04 Execution')

chunk_by_main_clause -  Split by-laws document into chunks based on main clause headings (e.g., 2.01). 
                         Section headers are treated as metadata state, not chunks.

ingest_bylaws_pdfs -  full pipeline: PDF paths → list of dicts chunks

"""


import re
from pypdf import PdfReader
from typing import List, Dict



### LOAD ###
def load_pdf_text(pdf_path: PATH) -> str:   
    """
    Extracts raw text from a PDF file.
    Returns all pages joined as
    single concatenated string.

    """

    reader = PdfReader(pdf_path) 
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


### NORMALIZE ###

def normalize_text(text: str) -> str:
    """
    Light cleanup only:
    - Strip trailing whitespace from each line
    - Preserves blank lines (needed for pattern matching)
    """

    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)



def normalize_leading_clause_whitespace(text: str) -> str:
    """
    Remove leading whitespace before clause numbers.
    e.g. '  2.04 Execution' → '2.04 Execution'
    This is necessary because PDF extraction often adds indent noise.
    """
    return re.sub(
        r"(?m)^\s+(?=\d+\.\d+)",
        "",
        text
    )



# REGEX PATTERNS FOR CLAUSE-AWARE CHUNKING AND CHUNKING FUNCTION

SECTION_HEADER_PATTERN = re.compile(
    r"""
    ^\s*                                             # leading whitespace / PDF noise
    (?P<section>SECTION\s+[A-Z][A-Z][^\n]{1,50})     #  e.g. "SECTION TWO" or "SECTION
    \s*                                              # optional space
    (?P<sec_title>[A-Z][A-Z\s]{3,100})               # section title (ALL CAPS)
    \s*$                                             # end of line
    """,
    re.MULTILINE | re.VERBOSE
)



# MAIN_CLAUSE_PATTERN
MAIN_CLAUSE_PATTERN = re.compile(
    r"""
    ^\s*                                           # leading whitespace from PDF extraction
    (?P<clause>\d+\.\d+)                           # depth-2 clause number only (e.g. 2.04, not 2.04.1)
    (?:\.)?                                        # Non-capture group - optional trailing period
    \s+
    (?P<cl_title>[A-Z][^:\n]{3,200})               # title: starts with capital, stops at colon or newline
    :\s*$                                          # COLON REQUIRED, and trailing white space to end of line
    """,
    re.MULTILINE | re.VERBOSE 
)




### CHUNK ###

def chunk_by_main_clause(text: str) -> List[Dict]:
   
    """
    Split the by-laws document into chunks based on main clause headings.
 
    Section headers (e.g. SECTION TWO BUSINESS OF THE COMPANY) are treated
    as metadata state — they label the chunks below them but are not chunks
    themselves.
 
    Returns a list of dicts, each with:
        {
            "text": str,          ← full clause text
            "metadata": {
                "section": str,
                "section_title": str,
                "clause_id": str,   ← e.g. "2.04"
                "clause_title": str,
                "doc_type": str
            }
        }
    """
    # Collect all structural markers
    markers = []

    for m in SECTION_HEADER_PATTERN.finditer(text):
        markers.append(("section", m))

    for m in MAIN_CLAUSE_PATTERN.finditer(text):
        markers.append(("clause", m))

    # Preserve document order
    markers.sort(key=lambda x: x[1].start())

    chunks = []

    current_section = {
        "section": None,
        "section_title": None
    }

    for i, (kind, match) in enumerate(markers):

        if kind == "section":
            # Update section state only
            current_section["section"] = match.group("section")
            current_section["section_title"] = match.group("sec_title")
            continue

        # kind == "clause"
        start = match.start()
        end = markers[i + 1][1].start() if i + 1 < len(markers) else len(text)

        clause_text = text[start:end].strip()

        chunks.append({
            "text": clause_text,
            "metadata": {
                "section": current_section["section"],
                "section_title": current_section["section_title"],
                "clause_id": match.group("clause"),
                "clause_title": match.group("cl_title"),
                "doc_type": "By-laws"
            }
        })

    return chunks




###  INGEST FULL PIPELINE ###

def ingest_bylaws_pdfs(pdf_paths: PATH) -> List[Dict]:
    """
    End-to-end ingestion: PDF file paths → list of clause chunks.
    Designed to accept a list so future by-law documents can be added easily.
    """

    all_chunks = []

    for pdf_path in pdf_paths:
        raw_text = load_pdf_text(pdf_path)
        norm_text = normalize_text(raw_text)
        clean_text = normalize_leading_clause_whitespace(norm_text)

        clause_chunks = chunk_by_main_clause(text=clean_text)
        for chunk in clause_chunks:
            chunk['metadata']['source'] = pdf_path.name
        all_chunks.extend(clause_chunks)

    return all_chunks