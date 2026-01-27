import streamlit as st
from pathlib import Path
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Speed WMS Chatbot",
    page_icon="💬",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🤖 Puks AI")
st.sidebar.markdown("Speed WMS Assistant")

options = ["💬 Chatbot", "🆘 Help & Support"]
page = st.sidebar.selectbox("Main Menu", options)

st.sidebar.divider()
st.sidebar.caption("© Speed WMS • AI Support(development phase) developed by Kgathola Puka")

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
    # CHATBOT THEME (CSS)
    # --------------------------------------------------
    st.markdown("""
    <style>
    div[data-testid="stChatMessage"][data-user="user"] {
        background-color: #0072C6 !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 10px 15px !important;
        max-width: 80% !important;
        margin-left: auto !important;
    }

    div[data-testid="stChatMessage"][data-user="assistant"] {
        background-color: #F0F2F6 !important;
        color: black !important;
        border-radius: 20px !important;
        padding: 10px 15px !important;
        max-width: 80% !important;
        margin-right: auto !important;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 15px !important;
        padding: 10px !important;
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)



    # --------------------------------------------------
    # LOAD VECTOR STORE
    # --------------------------------------------------
    @st.cache_resource
    def load_vector_store():
        vector_store_path = Path(
            r"C:\Users\kgathola.puka\OneDrive - MSC\Documents\GitHub\RCP(test)\SPEED CHATBOT PROJECT\DATA\vector_store"
        )
        index = faiss.read_index(str(vector_store_path / "faiss.index"))
        with open(vector_store_path / "metadata.pkl", "rb") as f:
            chunks = pickle.load(f)
        model = SentenceTransformer("all-mpnet-base-v2")
        return index, chunks, model

    index, chunks, embedding_model = load_vector_store()

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------
    def retrieve_context(query, top_k=5, max_distance=1.0):
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

        "You are a senior Speed WMS domain expert and trainer.\n"
                        "You must give VERY DETAILED, STEP-BY-STEP answers.\n"
                        "Rules:\n"
                        "- Use ONLY the provided context\n"
                        "- Explain each step clearly\n"
                        "- Use numbered steps and bullet points\n"
                        "- Include sub-steps where relevant\n"
                        "- If something is unclear, let them know that they can contact kgathola Puka for more questions or log a ticket\n"
                        "- If the answer is not in the context, say exactly: I do not know."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        return prompt

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    client = Groq(api_key="")

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
            max_tokens=1000,
            top_p=0.9
        )
        return completion.choices[0].message.content.strip()


    # --------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hi! I’m **Puks**, your Speed WMS assistant. How can I help?"
        })

    # --------------------------------------------------
    # DISPLAY CHAT
    # --------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --------------------------------------------------
    # INPUT
    # --------------------------------------------------
    user_input = st.chat_input("Ask a Speed WMS question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

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

# ==================================================
# 🆘 HELP & SUPPORT PAGE
# ==================================================
if page == "🆘 Help & Support":

    st.header("Welcome to Puks Help & Support 🆘")

    
    st.markdown("""
    #### 👋 You’re not stuck ####

    If Puks couldn’t fully answer your question or something doesn’t make sense,
    you can send your query to **human support**.

    🧠 In the future, this will create a **support ticket** automatically.
    """)

    st.info("💡 Tip: Include screenshots, steps, or exact error messages.")

    st.header("📬 Contact Support")

    with st.form("support_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Describe your issue in detail...")
        submitted = st.form_submit_button("Send to Support")

    if submitted:
        try:
            msg = EmailMessage()
            msg["Subject"] = "🆘 Speed WMS Chatbot Support Request"
            msg["From"] = "kgathola.puka@aglgroup.com"  
            msg["To"] = "kgathola.puka@aglgroup.com"    
            msg.set_content(f"""
Name: {name}
Email: {email}

Message:
{message}
""")

            # -----------------------------
            # Send via Outlook SMTP
            # -----------------------------
            server = smtplib.SMTP("smtp.office365.com", 587)
            server.starttls()
            server.login("kgathola.puka@aglgroup.com", "APP_PASSWORD")  # use your Outlook app password
            server.send_message(msg)
            server.quit()

            st.success("✅ Your message has been sent successfully!")
        except Exception as e:
            st.error(f"❌ Failed to send message. Error: {e}")

    st.markdown("""
    ---
    🤖 **Puks AI Assistant**  
    Built to help. Learning every day.
    """)
