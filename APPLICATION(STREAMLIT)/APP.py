import streamlit as st
from pathlib import Path
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from rank_bm25 import BM25Okapi
from collections import deque

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Puks AI(Predictive Unified Knowledge System)",
    page_icon="🚀",
    layout="wide"
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🤖 Puks AI")
st.sidebar.markdown("Enterprise Speed WMS Intelligence")

options = ["💬 Chatbot", "🆘 Help & Support"]
page = st.sidebar.selectbox("Navigation", options)

st.sidebar.divider()
st.sidebar.subheader("🧠 Model Settings")

AVAILABLE_MODELS = {
    " Llama 3.3 70B (Executive Mode – Best Overall)": "llama-3.3-70b-versatile",
    " Llama 4 Maverick 17B (Newest Gen)": "meta-llama/llama-4-maverick-17b-128e-instruct",
    " Qwen 3 32B (Structured Reasoning)": "qwen/qwen3-32b",
    "GPT-OSS 120B (Best Model)": "openai/gpt-oss-120b",
    " Llama 3.1 8B (Fast)": "llama-3.1-8b-instant"
}

selected_model_label = st.sidebar.selectbox(
    "Choose Model",
    list(AVAILABLE_MODELS.keys())
)

SELECTED_MODEL = AVAILABLE_MODELS[selected_model_label]
debug_mode = st.sidebar.toggle("🔍 Show Retrieved Context", value=False)

st.sidebar.success(f"Model Active: {selected_model_label}")
st.sidebar.divider()
st.sidebar.caption("© Puks AI System (Predictive Unified Knowledge System)")

# ==================================================
# LOAD VECTOR STORE & MODELS
# ==================================================
@st.cache_resource
@st.cache_resource
def load_resources():
    VECTOR_STORE = Path(__file__).parent / "data" / "vector_store"

    # ✅ Load prebuilt FAISS index
    index = faiss.read_index(str(VECTOR_STORE / "faiss.index"))

    # ✅ Load metadata (chunks)
    with open(VECTOR_STORE / "metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    # ✅ Load embedding model (only for query embedding)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Load reranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ✅ Build BM25 once
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return index, chunks, bm25, embedding_model, reranker


index, chunks, bm25, embedding_model, reranker = load_resources()

@st.cache_resource
def load_llm():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = load_llm()

# ==================================================
# MEMORY
# ==================================================
class ConversationMemory:
    def __init__(self, max_turns=8):
        self.history = deque(maxlen=max_turns)

    def add_user(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant(self, message):
        self.history.append({"role": "assistant", "content": message})

    def format(self):
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=8)

memory = st.session_state.memory


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def detect_document_type(chunk):
    structured = chunk.get("structured_data")
    if not structured:
        return "TEXT"
    if isinstance(structured, dict):
        if "columns" in structured:
            return "TABLE_SCHEMA"
        if "procedures" in structured or "core_tables" in structured:
            return "OPERATIONAL_REFERENCE"
    return "TEXT"

def validate_context(retrieved_chunks):
    if not retrieved_chunks:
        return False
    high_quality = [c for c in retrieved_chunks if c.get("final_score",0) > 0.18]
    return len(high_quality) > 0

def validate_answer(answer, retrieved_chunks):
    if not answer or len(answer.strip()) < 20:
        return False
    lower_answer = answer.lower()
    phrases = ["not mentioned", "not provided", "no information"]
    if any(p in lower_answer for p in phrases):
        return False
    return True

# ==================================================
# RETRIEVAL
# ==================================================
def retrieve_context(query, top_k=5):
    query_lower = query.lower()
    query_tokens = query_lower.split()

    schema_keywords = ["sql","select","query","join","where","insert","update","column","columns","table","schema","foreign key","primary key"]
    operational_keywords = ["reverse","reset","grn","receipt","shipment","mission","cancel","validate","close","reopen"]

    is_schema_query = any(k in query_lower for k in schema_keywords)
    is_operational_query = any(k in query_lower for k in operational_keywords)

    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, 40)
    bm25_scores = bm25.get_scores(query_tokens)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        metadata = chunk.get("metadata", {})
        doc_type = detect_document_type(chunk)

        vector_score = (float(score)+1)/2
        keyword_score = bm25_scores[idx]/8
        hybrid_score = (0.6*vector_score) + (0.3*keyword_score)

        if is_operational_query and doc_type == "OPERATIONAL_REFERENCE":
            hybrid_score += 0.4
        if is_schema_query and doc_type == "TABLE_SCHEMA":
            hybrid_score += 0.3

        table_name = metadata.get("table_name")
        if table_name and table_name.lower() in query_lower:
            hybrid_score += 0.5

        results.append({
            "score": hybrid_score,
            "doc_type": doc_type,
            "text": chunk["text"],
            "metadata": metadata,
            "structured_data": chunk.get("structured_data")
        })

    if not results:
        return [], 0.0

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:25]
    pairs = [(query, r["text"]) for r in results]
    rerank_scores = reranker.predict(pairs)

    for i, r in enumerate(results):
        r["rerank_score"] = float(rerank_scores[i])
        r["final_score"] = (0.7*r["score"]) + (0.3*r["rerank_score"])

    candidates = sorted(results, key=lambda x: x["final_score"], reverse=True)[:top_k]
    confidence = np.mean([r["final_score"] for r in candidates])
    return candidates, float(confidence)

# ==================================================
# PROMPT BUILDER
# ==================================================
def build_prompt(query, retrieved_chunks, memory_text="", sql_mode=False):

    context_sections = []
    has_schema = False

    # ----------------------------
    # EXPAND STRUCTURED DATA
    # ----------------------------
    for c in retrieved_chunks:
        metadata = c.get("metadata", {})
        text = c.get("text", "")

        structured = c.get("structured_data")

        # If this chunk contains schema JSON
        if isinstance(structured, dict) and "columns" in structured:
            has_schema = True

            table_name = structured.get("table_name", "UNKNOWN")
            description = structured.get("description", "No description available")
            primary_keys = structured.get("primary_key", [])

            table_text_lines = []
            table_text_lines.append(f"TABLE NAME: {table_name}")
            table_text_lines.append(f"DESCRIPTION: {description}")
            table_text_lines.append(f"PRIMARY KEY: {primary_keys}")
            table_text_lines.append("COLUMNS:")

            for col in structured.get("columns", []):
                column_block = [
                    f"Column Name: {col.get('name', 'UNKNOWN')}",
                    f"Description: {col.get('description', 'No description')}",
                    f"SQL Server Type: {col.get('type_sql_server', 'UNKNOWN')}",
                    f"Oracle Type: {col.get('type_oracle', 'UNKNOWN')}",
                    f"Is Primary Key: {col.get('is_primary_key', False)}",
                    f"Is Foreign Key: {col.get('is_foreign_key', False)}"
                ]

                if col.get("references_table") and col.get("references_column"):
                    column_block.append(
                        f"References: {col.get('references_table')}.{col.get('references_column')}"
                    )

                table_text_lines.append("\n".join(column_block))
                table_text_lines.append("-" * 40)

            text = "\n".join(table_text_lines)

        context_sections.append(
            f"[SOURCE: {metadata.get('source','unknown')} | "
            f"CATEGORY: {metadata.get('category','unknown')} | "
            f"TABLE: {metadata.get('table_name','N/A')}]\n{text}"
        )

    context_text = "\n\n".join(context_sections)

    # ----------------------------
    # DETECT SCHEMA QUESTIONS
    # ----------------------------
    schema_keywords = [
        "column", "columns", "schema", "structure",
        "table definition", "fields", "table structure"
    ]

    is_schema_question = any(word in query.lower() for word in schema_keywords)

    # ----------------------------
    # SCHEMA INSTRUCTIONS
    # ----------------------------
    schema_hint = ""
    if has_schema and is_schema_question:
        schema_hint = """
SCHEMA MODE ACTIVE.

If the user is asking about a table structure or columns:

You MUST provide a COMPLETE SCHEMA DEFINITION including:

1. Table Name
2. Table Description (if available)
3. Primary Key(s)
4. Total Number of Columns
5. A structured table listing ALL columns with:

   - Column Name
   - Description
   - SQL Server Data Type
   - Oracle Data Type
   - Primary Key (Yes/No)
   - Foreign Key (Yes/No)
   - Reference Table.Column (if applicable)

Do NOT return only column names.
Do NOT skip metadata.
Use ONLY the provided context.
If schema details are incomplete in context, state that clearly.
"""

    # ----------------------------
    # SQL MODE INSTRUCTIONS
    # ----------------------------
    sql_hint = ""
    if sql_mode:
        sql_hint = """
SQL MODE ACTIVE.

If SQL is required:

- Generate production-ready SQL
- Use explicit joins
- Do NOT use SELECT *
- Use proper formatting
- Only reference tables/columns found in the context
"""

    # ----------------------------
    # FINAL PROMPT
    # ----------------------------
    prompt = f"""
You are the Technical Architect of Speed WMS.

CRITICAL RULES:
- Answer strictly using the provided context.
- Do NOT invent columns, tables, or relationships.
- If the answer is not found in the context, respond:
  "I do not know, please contact support."

Conversation History:
{memory_text}

======================
CONTEXT
======================
{context_text}

======================
USER QUESTION
======================
{query}

======================
INSTRUCTIONS
======================
{schema_hint}
{sql_hint}

Provide a clear, professional, structured response:
"""

    return prompt.strip()


# ==================================================
# LLM CALL
# ==================================================
def get_llm_answer(prompt):
    try:
        completion = client.chat.completions.create(
            model=SELECTED_MODEL,
            messages=[
                {"role":"system","content":"You are a Speed WMS expert."},
                {"role":"user","content":prompt}
            ],
            max_tokens=3000,
            temperature=0,
            top_p = 1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LLM request failed: {str(e)}"

# ==================================================
# ASK FUNCTION
# ==================================================
def ask(question):

    sql_keywords = [
        "sql", "select", "join", "query","column", "columns", "schema", "structure",
        "table definition", "fields", "table structure",
        "code", "powerbi", "dashboard","ssrs",
        "how long", "count", "group by"
    ]

    sql_mode = any(k in question.lower() for k in sql_keywords)

    retrieved, confidence = retrieve_context(question)

    if not validate_context(retrieved):
        return "I do not know, please contact support.", retrieved

    prompt = build_prompt(
        query=question,
        retrieved_chunks=retrieved,
        memory_text=memory.format(),
        sql_mode=sql_mode   # 🔥 CRITICAL FIX
    )

    answer = get_llm_answer(prompt)

    if not validate_answer(answer, retrieved):
        answer = "I do not know, please contact support."

    memory.add_user(question)
    memory.add_assistant(answer)

    return answer, retrieved


# ==================================================
# CHAT UI
# ==================================================
if page == "💬 Chatbot":
    st.title("🚀 Puks AI")
    st.caption("Speed WMS Retrieval-Augmented Intelligence System")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Welcome. I am Puks — Speed WMS Retrieval-Augmented Intelligence System."
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about Speed WMS...")

    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing documentation..."):
                answer, retrieved = ask(user_input)
                if debug_mode:
                    with st.expander("🔍 Retrieved Context"):
                        for r in retrieved:
                            st.write("Final Score:", round(r.get("final_score",0),3))
                            st.write("Metadata:", r["metadata"])
                            st.write(r["text"][:600])
                            st.divider()


            st.markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})

# ==================================================
# HELP PAGE
# ==================================================
if page == "🆘 Help & Support":
    st.header("🆘 Help & Support")
    with st.form("support_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        issue = st.text_area("Describe your issue")
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.success("✅ Support request captured.")





