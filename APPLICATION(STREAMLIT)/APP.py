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
    "🔥 Llama 3.3 70B (Executive Mode – Best Overall)": "llama-3.3-70b-versatile",
    "🚀 Llama 4 Maverick 17B (Newest Gen)": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "🧠 Qwen 3 32B (Structured Reasoning)": "qwen/qwen3-32b",
    "💎 GPT-OSS 120B (Massive Model)": "openai/gpt-oss-120b",
    "⚡ Llama 3.1 8B (Fast)": "llama-3.1-8b-instant"
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
# LOAD VECTOR STORE (YOUR DESKTOP LOGIC)
# ==================================================
@st.cache_resource
def load_resources():

    VECTOR_STORE = Path(__file__).parent / "data" / "vector_store"

    index = faiss.read_index(str(VECTOR_STORE / "faiss.index"))

    with open(VECTOR_STORE / "metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Embedding model (DESKTOP VERSION)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Rebuild FAISS index (LIKE DESKTOP)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(embeddings)

    # BM25
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return new_index, chunks, bm25, embedding_model, reranker


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
        self.history = deque(maxlen=max_turns*2)

    def add_user(self, msg):
        self.history.append({"role": "user", "content": msg})

    def add_assistant(self, msg):
        self.history.append({"role": "assistant", "content": msg})

    def format(self):
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)

memory = ConversationMemory()

# ==================================================
# RETRIEVAL (YOUR DESKTOP VERSION)
# ==================================================
def retrieve_context(query, top_k=3):
    query_lower = query.lower()
    query_tokens = query_lower.split()
    schema_keywords = ["sql","select","query","join","where","insert","update","column","table","queries","relational","foreign key","primary key"]
    is_schema_query = any(word in query_lower for word in schema_keywords)

    # Vector search
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
    scores, indices = index.search(query_embedding, 30)

    # BM25 scoring
    bm25_scores = bm25.get_scores(query_tokens)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        metadata = chunk.get("metadata", {})
        text = chunk["text"]

        vector_score = float(score)
        keyword_score = bm25_scores[idx] / 10
        hybrid_score = vector_score + keyword_score

        if is_schema_query and metadata.get("is_table_schema", False):
            hybrid_score += 0.15

        table_name = metadata.get("table_name")
        if table_name and table_name.lower() in query_lower:
            hybrid_score += 0.4

        results.append({
            "score": hybrid_score,
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "text": text,
            "metadata": metadata,
            "structured_data": chunk.get("structured_data")  # store full table JSON if present
        })

    if not results:
        return []

    # Sort and rerank
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    candidates = results[:20]
    pairs = [(query, r["text"]) for r in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, r in enumerate(candidates):
        r["rerank_score"] = float(rerank_scores[i])

    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ==================================================
# PROMPT BUILDER (FULL ARCHITECT MODE)
# ==================================================
def build_prompt(query, retrieved_chunks, memory_text="", sql_mode=False):

    context_sections = []

    for c in retrieved_chunks:
        metadata = c.get("metadata", {})
        text = c["text"]

        # If structured table JSON exists → build full readable schema
        if c.get("structured_data"):
            table_json = c["structured_data"]

            table_text_lines = [
                f"TABLE NAME: {table_json.get('table_name')}",
                f"DESCRIPTION: {table_json.get('description','No description')}",
                f"PRIMARY KEY: {table_json.get('primary_key',[])}",
                "COLUMNS:"
            ]

            for col in table_json.get("columns", []):
                line = f"- {col.get('name','UNKNOWN')}: {col.get('description','')}"
                line += f" | SQL Type: {col.get('type_sql_server','UNKNOWN')}"
                line += f", Oracle Type: {col.get('type_oracle','UNKNOWN')}"
                line += f" | PK: {col.get('is_primary_key',False)}"
                line += f", FK: {col.get('is_foreign_key',False)}"

                if col.get('references_table') or col.get('references_column'):
                    line += f" | References: {col.get('references_table')}.{col.get('references_column')}"

                table_text_lines.append(line)

            text = "\n".join(table_text_lines)

        context_sections.append(
            f"[SOURCE: {metadata.get('source','unknown')} | "
            f"CATEGORY: {metadata.get('category','unknown')} | "
            f"TABLE: {metadata.get('table_name','N/A')} | "
            f"SCORE: {c.get('score',0):.3f}]\n{text}"
        )

    context_text = "\n\n".join(context_sections)

    # Detect context types
    has_schema = any(
        c.get("metadata", {}).get("is_table_schema", False)
        for c in retrieved_chunks
    )

    has_process_docs = any(
        c.get("metadata", {}).get("category", "").lower() not in ["database", "schema"]
        for c in retrieved_chunks
    )

    # ===== BEHAVIOR INSTRUCTIONS =====

    behavior_instructions = """
You are the creator and technical architect of Speed WMS.

You understand:
- Database schema and relationships
- Business logic behind tables
- How receipts are created
- How picking works
- How movements are generated
- Configuration steps
- User workflows in the UI
- Troubleshooting operational issues
- SQL queries when needed

Answer based strictly on the provided context.
If multiple sources exist, combine them intelligently.
Be clear, structured, and authoritative.
"""

    schema_instruction = ""
    if has_schema:
        schema_instruction = """
If the user is asking about a table or columns:
- List ALL columns found in the context.
- Format as:
  Table Name | Column Name | Description | Type | PK/FK | References
- Do NOT invent columns.
- If helpful, include example SELECT statements.
"""

    operational_instruction = ""
    if has_process_docs:
        operational_instruction = """
If the question is about a process (e.g., creating a receipt, picking, configuration):
- Provide step-by-step instructions.
- Explain what happens in the system.
- Mention related tables only if relevant.
- Explain business impact where useful.
"""

    sql_instruction = ""
    if sql_mode:
        sql_instruction = """
If the question is SQL-related:
- Provide clean, production-ready SQL examples.
- Use proper joins and filtering when relevant.
"""

    prompt = f"""
{behavior_instructions}

Rules:
- Use ONLY the provided context.
- Do NOT hallucinate.
- If the answer is not found in context, say exactly:
  'I do not know, please contact the support team or submit a ticket.'

Conversation so far:
{memory_text}

Context:
{context_text}

User Question:
{query}

Instructions:
{schema_instruction}
{operational_instruction}
{sql_instruction}

Answer clearly and professionally:
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
                {"role": "system", "content": "You are a Speed WMS expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ LLM request failed: {str(e)}"

# ==================================================
# ASK
# ==================================================
def ask(question):

    retrieved = retrieve_context(question)

    prompt = build_prompt(
        question,
        retrieved,
        memory.format()
    )

    answer = get_llm_answer(prompt)

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
                            st.write(r["metadata"])
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











