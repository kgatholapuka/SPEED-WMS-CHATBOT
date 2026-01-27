import streamlit as st
from pathlib import Path
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
import datetime
from datetime import datetime
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Speed WMS Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Speed WMS Chatbot")

st.markdown("""
**Speed WMS Chatbot** answers questions strictly based on  
Speed WMS documentation using Retrieval-Augmented Generation (RAG).

⚠️ If the information is not found in the documentation, the assistant will say  
**"I do not know."**

""")

st.divider()

# --------------------------------------------------
# CHATBOT THEME (CSS)
# --------------------------------------------------
st.markdown("""
<style>
/* User bubble */
div[data-testid="stChatMessage"][data-user="user"] {
    background-color: #0072C6 !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 10px 15px !important;
    max-width: 80% !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}

/* Assistant bubble */
div[data-testid="stChatMessage"][data-user="assistant"] {
    background-color: #F0F2F6 !important;
    color: black !important;
    border-radius: 20px !important;
    padding: 10px 15px !important;
    max-width: 80% !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}

/* Recent question buttons */
.recent-question-button button {
    background-color: #E1ECF4 !important;
    color: #0072C6 !important;
    border-radius: 10px !important;
    border: 1px solid #0072C6 !important;
    padding: 5px 10px !important;
    margin-right: 5px !important;
    margin-bottom: 5px !important;
    font-weight: bold !important;
}

/* Chat input */
div[data-testid="stChatInput"] > div > textarea {
    border-radius: 15px !important;
    border: 1px solid #ccc !important;
    padding: 10px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Chat Settings")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05
)

top_k = st.sidebar.slider(
    "Top-K Tokens",
    min_value=10,
    max_value=100,
    value=40,
    step=5
)

max_tokens = st.sidebar.slider(
    "Max Response Tokens",
    min_value=100,
    max_value=800,
    value=350,
    step=50
)

st.sidebar.divider()

# --------------------------------------------------
# CHAT RESET BUTTON
# --------------------------------------------------
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.session_state.recent_questions = []
    st.sidebar.success("Chat history cleared")

# --------------------------------------------------
# EXPORT CHAT
# --------------------------------------------------
def export_chat(messages):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    content = []

    for msg in messages:
        role = msg["role"].upper()
        content.append(f"{role}:\n{msg['content']}\n")

    return "\n".join(content), f"speed_wms_chat_{timestamp}.txt"


if st.session_state.get("messages"):
    chat_text, filename = export_chat(st.session_state.messages)

    st.sidebar.download_button(
        label="📥 Export Chat",
        data=chat_text,
        file_name=filename,
        mime="text/plain"
    )

# --------------------------------------------------
# LOAD VECTOR STORE (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_vector_store():
    vector_store_path = Path(r"C:\Users\kgathola.puka\OneDrive - MSC\Documents\GitHub\RCP(test)\SPEED CHATBOT PROJECT\DATA\vector_store")

    index = faiss.read_index(str(vector_store_path / "faiss.index"))

    with open(vector_store_path / "metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    return index, chunks, embedding_model

index, chunks, embedding_model = load_vector_store()

# --------------------------------------------------
# RETRIEVAL FUNCTION
# --------------------------------------------------
def retrieve_context(query, top_k=6, max_distance=1.2):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = np.array(query_embedding).astype("float32")

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension ({query_embedding.shape[1]}) "
            f"does not match FAISS index dimension ({index.d})."
        )

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if dist <= max_distance:
            results.append(chunks[idx]["text"])

    return results

# --------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------
def build_prompt(context_chunks, question):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a domain expert assistant for Speed WMS.

Rules:
- Use ONLY the information provided in the context
- Answer clearly using bullet points or numbered steps
- Do NOT add external knowledge
- If the answer is not in the context, say exactly: "I do not know."

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

# --------------------------------------------------
# LLM ANSWER (HYPER-TUNED)
# --------------------------------------------------
client = Groq(api_key="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")

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
                    "- Use ONLY the provided context\n"
                    "- Explain each step clearly\n"
                    "- Use numbered steps and bullet points\n"
                    "- Include sub-steps where relevant\n"
                    "- If something is unclear, let them know that they can contact kgathola Puka for more questions or log a ticket\n"
                    "- If the answer is not in the context, say exactly: I do not know."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=450,
        top_p=0.9
    )
    return completion.choices[0].message.content.strip()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []

# --------------------------------------------------
# WELCOME MESSAGE
# --------------------------------------------------
if not st.session_state.messages:
    welcome_text = (
        "💬 Hello! I’m Puks (Predictive Unified Knowledge System) your AI assistant for Speed WMS.\n\n"
        "Ask me anything about Speed WMS documentation and I’ll guide you step by step.\n"
        
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_text})

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# SHOW RECENT QUESTIONS (STYLED)
# --------------------------------------------------
# USER INPUT + RAG EXECUTION
# --------------------------------------------------
user_input = st.chat_input("Ask a question about Speed WMS...")

if user_input:
    # Add user message with emoji
    st.session_state.messages.append({"role": "user", "content": f"🧑 {user_input}"})

    # Update recent questions (max 5)
    if user_input not in st.session_state.recent_questions:
        st.session_state.recent_questions.insert(0, user_input)
        st.session_state.recent_questions = st.session_state.recent_questions[:5]

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation...", show_time=True):
            st.write("🔍 Retrieving relevant information...")
            retrieved_chunks = retrieve_context(user_input)
            st.success(f"Found {len(retrieved_chunks)} relevant chunks.")
            st.write("🧱 Building prompt...")
            st.success("Prompt built.")
            prompt = build_prompt(retrieved_chunks, user_input)
            st.write("🤖 Generating answer...")
            answer = get_llm_answer(prompt)
            st.success("Answer generated.")

        # Append assistant response with emoji
        answer_with_emoji = f"💬 {answer}"
        st.markdown(answer_with_emoji)
        st.session_state.messages.append({"role": "assistant", "content": answer_with_emoji})

        # Sources
        with st.expander("📄 Sources used"):
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Source {i+1}:** {chunk[:1000]}...")
