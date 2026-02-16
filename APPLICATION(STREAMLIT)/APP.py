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
    page_title="Speed WMS Chatbot",
    page_icon="💬",
    layout="wide"
)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("🤖 Puks AI")
st.sidebar.markdown("Speed WMS Assistant")

options = ["💬 Chatbot", "🆘 Help & Support"]
page = st.sidebar.selectbox("Main Menu", options)

st.sidebar.divider()
st.sidebar.caption(
    "© Speed WMS • AI Support (Development Phase)\n"
    "Developed by Kgathola Puka"
)

# ==================================================
# LOAD RESOURCES (CACHED)
# ==================================================
@st.cache_resource
def load_vector_store():
    VECTOR_STORE = Path(__file__).parent / "data" / "vector_store"

    index = faiss.read_index(str(VECTOR_STORE / "faiss.index"))

    with open(VECTOR_STORE / "metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)

    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return index, chunks, bm25, embedding_model, reranker

@st.cache_resource
def load_llm_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

index, chunks, bm25, embedding_model, reranker = load_vector_store()
client = load_llm_client()

# ==================================================
# SESSION-BASED CONVERSATION MEMORY
# ==================================================
MAX_TURNS = 8

class ConversationMemory:
    def __init__(self, max_turns=MAX_TURNS):
        self.history = deque(maxlen=max_turns*2)  # user+assistant

    def add_user(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant(self, message):
        self.history.append({"role": "assistant", "content": message})

    def format(self):
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)

memory = ConversationMemory()

# ==================================================
# RETRIEVAL FUNCTION (Vector + BM25 + CrossEncoder)
# ==================================================
def retrieve_context(query, top_k=8):
    query_lower = query.lower()
    query_tokens = query_lower.split()

    schema_keywords = [
        "sql", "select", "query", "join", "where", "insert", "update",
        "column", "table", "queries", "relational", "foreign key", "primary key"
    ]
    is_schema_query = any(word in query_lower for word in schema_keywords)

    # Vector search
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
    scores, indices = index.search(query_embedding, 30)

    # BM25
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
            "metadata": metadata
        })

    if not results:
        return []

    # Rerank
    candidates = sorted(results, key=lambda x: x["score"], reverse=True)[:20]
    pairs = [(query, r["text"]) for r in candidates]
    rerank_scores = reranker.predict(pairs)

    for i, r in enumerate(candidates):
        r["rerank_score"] = float(rerank_scores[i])

    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]

# ==================================================
# PROMPT BUILDER
# ==================================================
def build_prompt(query, retrieved_chunks, memory_text="", sql_mode=False):
    context_sections = []
    for c in retrieved_chunks:
        metadata = c.get("metadata", {})
        text = c["text"]

        table_name = metadata.get("table_name") or metadata.get("source", "N/A")
        category = metadata.get("category", "unknown")

        context_sections.append(
            f"[SOURCE: {metadata.get('source','unknown')} | "
            f"CATEGORY: {category} | TABLE: {table_name} | SCORE: {c.get('score',0):.3f}]\n{text}"
        )

    context_text = "\n\n".join(context_sections)

    table_chunks_present = any(c.get("metadata", {}).get("is_table_schema", False) for c in retrieved_chunks)

    structured_instruction = ""
    if table_chunks_present:
        structured_instruction = (
            "- List all table columns found in the context in a readable table format.\n"
            "- Include: Table Name | Column Name | Description.\n"
            "- Optionally, include example SQL SELECT statements.\n"
            "- Do NOT invent tables or columns not present in the context.\n"
        )

    sql_instructions = ""
    if sql_mode:
        sql_instructions = "- This query appears SQL-related; include sample SELECT statements where relevant.\n"

    prompt = f"""
You are an internal Speed WMS support assistant.

Rules:
- Use ONLY the provided context
- Answer in detailed, step-by-step format
{structured_instruction}{sql_instructions}
- If the answer is not found, say exactly: 'I do not know, please contact the support team or submit a ticket.'

Conversation so far:
{memory_text}

Context:
{context_text}

User question:
{query}

Answer (plain text, readable table if needed):
"""
    return prompt.strip()

# ==================================================
# LLM CALL
# ==================================================
def get_llm_answer(prompt):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":"You are a Speed WMS expert."},
                {"role":"user","content":prompt}
            ],
            max_tokens=2000,
            temperature=0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LLM request failed: {str(e)}"

# ==================================================
# CHAT HELPER
# ==================================================
def ask(question):
    retrieved_chunks = retrieve_context(question)
    sql_keywords = ["sql", "select", "query", "join", "where", "insert", "update", "column", "table"]
    is_sql_query = any(word in question.lower() for word in sql_keywords)

    prompt = build_prompt(
        query=question,
        retrieved_chunks=retrieved_chunks,
        memory_text=memory.format(),
        sql_mode=is_sql_query
    )

    answer = get_llm_answer(prompt)
    memory.add_user(question)
    memory.add_assistant(answer)

    return answer

# ==================================================
# 💬 CHATBOT PAGE
# ==================================================
if page == "💬 Chatbot":
    st.title("💬 Speed WMS Chatbot")
    st.markdown("""
**Puks** answers questions strictly based on  
**Speed WMS documentation** using Retrieval-Augmented Generation (RAG).

⚠️ If the information is not found, Puks will say:  
**“I do not know, please contact the support team or submit a ticket.”**
""")

    # Initialize session messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hi! I’m **Puks**, your Speed WMS assistant. How can I help?"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a Speed WMS question...")

    if user_input:
        st.session_state.messages.append({"role":"user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documentation..."):
                answer = ask(user_input)
            st.markdown(f"💬 {answer}")
            st.session_state.messages.append({"role":"assistant", "content": answer})

# ==================================================
# 🆘 HELP & SUPPORT PAGE
# ==================================================
if page == "🆘 Help & Support":
    st.header("🆘 Help & Support")

    st.markdown("""
If Puks could not fully answer your question, please log a support request.

📌 **Note:**  
Email sending is disabled on Streamlit Cloud.  
This section is ready for **Power Automate / Ticketing integration**.
""")

    with st.form("support_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        issue = st.text_area("Describe your issue in detail")
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.success(
            "✅ Support request captured.\n\n"
            "This will be connected to Power Automate in the next phase."
        )

    st.markdown("""
---
🤖 **Puks AI Assistant**  
Built to help. Learning every day.
""")
