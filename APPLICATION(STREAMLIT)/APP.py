import streamlit as st
from pathlib import Path
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

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
    base_path = Path(__file__).parent
    vector_store_path = base_path / "data" / "vector_store"

    index = faiss.read_index(str(vector_store_path / "faiss.index"))

    with open(vector_store_path / "metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer("all-mpnet-base-v2")
    return index, chunks, model


@st.cache_resource
def load_llm_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


index, chunks, embedding_model = load_vector_store()
client = load_llm_client()

# ==================================================
# SESSION-BASED CONVERSATION MEMORY
# ==================================================
MAX_TURNS = 8

def add_to_memory(role, content):
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []

    st.session_state.conversation_memory.append({
        "role": role,
        "content": content
    })

    # Keep only last N turns (user + assistant)
    if len(st.session_state.conversation_memory) > MAX_TURNS * 2:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-MAX_TURNS * 2:]


def format_memory():
    if "conversation_memory" not in st.session_state:
        return ""

    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.conversation_memory
    )

# ==================================================
# RETRIEVAL FUNCTION
# ==================================================
def retrieve_context(query, top_k=7, max_distance=1.0):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if dist <= max_distance:
            results.append(chunks[idx]["text"])

    return results

# ==================================================
# PROMPT BUILDER (GEN RAG + MEMORY)
# ==================================================
def build_prompt(question, retrieved_chunks, memory_text):
    context = "\n\n".join(f"[CONTEXT]\n{chunk}" for chunk in retrieved_chunks)

    return f"""
You are an internal Speed WMS support assistant.

Rules:
- Use ONLY the provided context
- Answer in VERY DETAILED, step-by-step format
- Use numbered steps and sub-steps
- If the answer is not found, say exactly: I do not know.

Conversation so far:
{memory_text}

Context:
{context}

User question:
{question}

Answer:
""".strip()

# ==================================================
# LLM CALL
# ==================================================
def get_llm_answer(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Speed WMS domain expert and trainer.\n"
                    "You must give VERY DETAILED, STEP-BY-STEP answers.\n"
                    "Rules:\n"
                    "- When a answer is based on database tables, include table name and all columns found\n"
                    "- Use ONLY the provided context\n"
                    "- Explain each step clearly\n"
                    "- Use numbered steps and bullet points\n"
                    "- Include sub-steps where relevant\n"
                    "- If unclear, advise contacting Kgathola Puka or logging a support ticket\n"
                    "- If answer not in context, say exactly: I do not know."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000,
        top_p=0.9
    )
    return completion.choices[0].message.content.strip()

# ==================================================
# 💬 CHATBOT PAGE
# ==================================================
if page == "💬 Chatbot":
    st.title("💬 Speed WMS Chatbot")

    st.markdown("""
**Puks** answers questions strictly based on  
**Speed WMS documentation** using Retrieval-Augmented Generation (RAG).

⚠️ If the information is not found, Puks will say:  
**“I do not know.”**
""")

    # --------------------------------------------------
    # CHAT UI STYLES
    # --------------------------------------------------
    st.markdown("""
    <style>
    div[data-testid="stChatMessage"][data-user="user"] {
        background-color: #0072C6;
        color: white;
        border-radius: 20px;
        padding: 10px 15px;
        max-width: 80%;
        margin-left: auto;
    }

    div[data-testid="stChatMessage"][data-user="assistant"] {
        background-color: #F0F2F6;
        color: black;
        border-radius: 20px;
        padding: 10px 15px;
        max-width: 80%;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # SESSION STATE INIT
    # --------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hi! I’m **Puks**, your Speed WMS assistant. How can I help?"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --------------------------------------------------
    # USER INPUT
    # --------------------------------------------------
    user_input = st.chat_input("Ask a Speed WMS question...")

    if user_input:
        # Add user to session memory
        add_to_memory("user", user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documentation..."):
                retrieved_chunks = retrieve_context(user_input)

                memory_text = format_memory()
                prompt = build_prompt(user_input, retrieved_chunks, memory_text)
                answer = get_llm_answer(prompt)

            final_answer = f"💬 {answer}"
            st.markdown(final_answer)

            # Add assistant answer to memory
            add_to_memory("assistant", answer)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # Sources used
            with st.expander("📄 Sources used"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i+1}:** {chunk[:1000]}...")

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
