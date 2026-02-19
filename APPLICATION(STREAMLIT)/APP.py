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
    page_title="Speed WMS AI",
    page_icon="🚀",
    layout="wide"
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🤖 Puks AI")
st.sidebar.markdown("Enterprise Speed WMS Assistant")

options = ["💬 Chatbot", "🆘 Help & Support"]
page = st.sidebar.selectbox("Navigation", options)

st.sidebar.divider()
st.sidebar.subheader("🧠 AI Settings")

AVAILABLE_MODELS = {
    "🔥 Llama 3.3 70B (Executive Mode – Best Overall)": "llama-3.3-70b-versatile",
    "🚀 Llama 4 Maverick 17B (Newest Gen)": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "🧠 Qwen 3 32B (Structured Reasoning)": "qwen/qwen3-32b",
    "💎 GPT-OSS 120B (Massive Model)": "openai/gpt-oss-120b",
    "⚖️ Llama 3.3 13B (Balanced)": "llama-3.3-13b-versatile",
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
st.sidebar.caption(
    "© Speed WMS • AI Support System\n"
    "Developed by Kgathola Puka"
)

# ==================================================
# LOAD RESOURCES
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
# MEMORY
# ==================================================
MAX_TURNS = 8

class ConversationMemory:
    def __init__(self, max_turns=MAX_TURNS):
        self.history = deque(maxlen=max_turns*2)

    def add_user(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant(self, message):
        self.history.append({"role": "assistant", "content": message})

    def format(self):
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)

memory = ConversationMemory()

# ==================================================
# RETRIEVAL
# ==================================================
def retrieve_context(query, top_k=8):
    query_lower = query.lower()
    query_tokens = query_lower.split()

    schema_keywords = [
        "sql","select","query","join","where",
        "insert","update","column","table",
        "foreign key","primary key"
    ]

    process_keywords = [
        "receipt","picking","movement","create",
        "config","setup","process"
    ]

    is_schema_query = any(word in query_lower for word in schema_keywords)
    is_process_query = any(word in query_lower for word in process_keywords)

    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
    scores, indices = index.search(query_embedding, 30)

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
            hybrid_score += 0.2

        if is_process_query and metadata.get("category","").lower() not in ["database","schema"]:
            hybrid_score += 0.2

        table_name = metadata.get("table_name")
        if table_name and table_name.lower() in query_lower:
            hybrid_score += 0.4

        results.append({
            "score": hybrid_score,
            "text": text,
            "metadata": metadata
        })

    if not results:
        return []

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
MAX_CONTEXT_CHARS = 12000

def build_prompt(query, retrieved_chunks, memory_text="", sql_mode=False):

    context_sections = []
    total_chars = 0

    for c in retrieved_chunks:
        metadata = c.get("metadata", {})
        text = c["text"]

        section = (
            f"[SOURCE: {metadata.get('source','unknown')} | "
            f"CATEGORY: {metadata.get('category','unknown')} | "
            f"TABLE: {metadata.get('table_name','N/A')}]\n{text}\n"
        )

        total_chars += len(section)
        if total_chars > MAX_CONTEXT_CHARS:
            break

        context_sections.append(section)

    context_text = "\n\n".join(context_sections)

    prompt = f"""
You are the creator and architect of Speed WMS.

You understand:
- Database schema
- Business logic
- Receipts
- Picking
- Movements
- Configuration
- Troubleshooting
- SQL

Use ONLY the provided context.
Do NOT hallucinate.
If answer not found say exactly:
'I do not know, please contact the support team or submit a ticket.'

Conversation:
{memory_text}

Context:
{context_text}

User Question:
{query}

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
                {"role": "system", "content": "You are the architect of Speed WMS."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ LLM request failed: {str(e)}"

# ==================================================
# ASK FUNCTION
# ==================================================
def ask(question):
    retrieved = retrieve_context(question)

    sql_keywords = ["sql","select","query","join","where","insert","update"]
    is_sql_query = any(word in question.lower() for word in sql_keywords)

    prompt = build_prompt(
        question,
        retrieved,
        memory.format(),
        sql_mode=is_sql_query
    )

    answer = get_llm_answer(prompt)

    memory.add_user(question)
    memory.add_assistant(answer)

    return answer, retrieved

# ==================================================
# CHATBOT PAGE
# ==================================================
if page == "💬 Chatbot":
    st.title("🚀 Speed WMS AI Assistant")
    st.caption("Enterprise Retrieval-Augmented Intelligence System")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Welcome. I am **Puks**, architect-level Speed WMS intelligence."
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
                            st.write(r["text"][:500])
                            st.divider()

            st.markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})

# ==================================================
# HELP PAGE
# ==================================================
if page == "🆘 Help & Support":
    st.header("🆘 Help & Support")

    st.markdown("""
If the AI could not fully answer your question,
please submit a support request.
""")

    with st.form("support_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        issue = st.text_area("Describe your issue")
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.success("✅ Support request captured.")

