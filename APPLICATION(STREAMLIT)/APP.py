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
st.sidebar.subheader("🧠 AI Settings")

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
st.sidebar.caption("© Speed WMS • AI Support System")

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

    schema_keywords = [
        "sql","select","query","join","where","insert","update",
        "column","table","foreign key","primary key"
    ]

    is_schema_query = any(word in query_lower for word in schema_keywords)

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)

    scores, indices = index.search(query_embedding, 30)

    bm25_scores = bm25.get_scores(query_tokens)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        metadata = chunk.get("metadata", {})

        vector_score = float(score)
        keyword_score = bm25_scores[idx] / 10
        hybrid_score = vector_score + keyword_score

        if is_schema_query and metadata.get("is_table_schema", False):
            hybrid_score += 0.2

        table_name = metadata.get("table_name")
        if table_name and table_name.lower() in query_lower:
            hybrid_score += 0.4

        results.append({
            "score": hybrid_score,
            "text": chunk["text"],
            "metadata": metadata,
            "structured_data": chunk.get("structured_data")
        })

    if not results:
        return []

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
def build_prompt(query, retrieved_chunks, memory_text=""):

    query_lower = query.lower()

    # ===============================
    # INTENT DETECTION
    # ===============================

    is_table_request = any(
        phrase in query_lower for phrase in [
            "all columns",
            "table",
            "schema",
            "structure",
            "explain table"
        ]
    )

    is_sql_generation = any(
        phrase in query_lower for phrase in [
            "sql",
            "select",
            "write query",
            "generate query",
            "how do i get",
            "how do i join",
            "report code",
            "join"
        ]
    )

    # ===============================
    # BUILD CONTEXT
    # ===============================

    context_sections = []

    for c in retrieved_chunks:

        text = c["text"]

        if c.get("structured_data"):

            table_json = c["structured_data"]

            lines = [
                f"TABLE NAME: {table_json.get('table_name')}",
                f"DESCRIPTION: {table_json.get('description','')}",
                f"PRIMARY KEY: {table_json.get('primary_key',[])}",
                "",
                "COLUMNS:"
            ]

            for col in table_json.get("columns", []):
                line = (
                    f"- {col.get('name')} "
                    f"(SQL Server: {col.get('type_sql_server','')}, "
                    f"Oracle: {col.get('type_oracle','')})"
                    f"\n    Description: {col.get('description','')}"
                    f"\n    PK: {col.get('is_primary_key',False)}"
                    f"\n    FK: {col.get('is_foreign_key',False)}"
                )

                if col.get("references_table"):
                    line += (
                        f"\n    References: "
                        f"{col.get('references_table')}."
                        f"{col.get('references_column')}"
                    )

                lines.append(line)

            text = "\n".join(lines)

        context_sections.append(text)

    context_text = "\n\n".join(context_sections)

    # ===============================
    # SYSTEM INSTRUCTIONS
    # ===============================

    if is_sql_generation:

        system_instruction = """
You are the original database architect of Speed WMS.

You MUST generate production-grade SQL.

JOIN RULES:

Inbound:
- Use NOSU (Support number) but the support number somes from the rel_dat and is not in ree_dat so when they want to use the recption header with the lines they have to use the ree_nore and the act_code
- Use REE_NORE (Reception header number)

Outbound:
- Use OPL_NOSU (Support number)
- Use OIPE_NOOE (Outbound line)

Tables commonly linked:
ree_dat, rel_dat, stk_dat, mvt_dat,ope_dat,opl_dat,mie_dat,mil_dat,chg_dat,chl_dat

Rules:
1. Always use correct join keys before using those keys be sure to check from the table schema what that column is in that table if the column is a FK key please understand the tables first so that when you join will make sure to use the best columns to join.
2. Prefer nosu to join buit also be carefully because nosu can be joined on the lines table e.g.rel_dat,opl_dat,mil_dat but can't be join from the headers. you can also search for more docuemnts to tell you about the support so that you can understand well.
3. Always use the best joins based on the question and the goal of the output.
4. Use clear aliases.
5. Provide complete working SQL.
6. Explain what the query does after the SQL.

Do NOT hallucinate columns.
Only use provided schema.
"""

    elif is_table_request:

        system_instruction = """
You are the original database architect of Speed WMS.

When explaining a table you MUST:

1. Explain the business purpose.
2. Provide full schema with meaning.
3. Explain PK and FK.
4. Explain inbound join logic.
5. Explain outbound join logic.
6. Provide example SQL joins.
7. Mention NOSU / REE_NORE for inbound.
8. Mention OPL_NOSU / OIPE_NOOE for outbound.

Be technical and structured.
"""

    else:

        system_instruction = """
You are a Speed WMS expert.
Use only provided context.
If not found say:
'I do not know, please contact the support team or submit a ticket.'
"""

    # ===============================
    # FINAL PROMPT
    # ===============================

    prompt = f"""
{system_instruction}

Conversation:
{memory_text}

Context:
{context_text}

User Question:
{query}

Respond clearly and professionally.
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
            max_tokens=2000,
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

    st.title("🚀 Speed WMS AI Assistant")
    st.caption("Enterprise Retrieval-Augmented Intelligence System")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Welcome. I am Puks — architect-level Speed WMS intelligence."
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




