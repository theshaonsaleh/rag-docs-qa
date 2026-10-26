"""
Streamlit UI for the RAG Q&A system.
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from rag_system import build_rag_system


# Page config
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="centered"
)

# Load environment
load_dotenv()


@st.cache_resource
def load_rag_system():
    """Load RAG system (cached)."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    with st.spinner('Loading RAG system...'):
        rag = build_rag_system(
            docs_dir=Path('sample_docs'),
            index_dir=Path('vector_index'),
            api_key=api_key,
            force_rebuild=False
        )
    return rag, bool(api_key)


# Title and description
st.title("📚 RAG Q&A System")
st.markdown("**Ask questions about the Riverside Public Library**")

# Load RAG system
rag, has_api_key = load_rag_system()

# Mode indicator
mode = "🤖 OpenAI Mode" if has_api_key else "🔧 Offline Mode"
st.sidebar.title("System Info")
st.sidebar.info(f"**Mode:** {mode}")

if not has_api_key:
    st.sidebar.warning(
        "Running in offline mode. Add OPENAI_API_KEY to .env for better answers."
    )

# Settings
st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 5, 3)

# Example questions
st.sidebar.subheader("Example Questions")
examples = [
    "What are the library hours?",
    "How do I get a library card?",
    "Can I use the meeting rooms?",
    "What digital resources are available?",
    "Is there a story time for kids?",
    "What is the printing cost?"
]

for example in examples:
    if st.sidebar.button(example, key=example):
        st.session_state['question'] = example

# Main interface
st.markdown("---")

# Question input
question = st.text_input(
    "Your question:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., What are the library hours?",
    key='question_input'
)

# Answer button
if st.button("🔍 Get Answer", type="primary") or question:
    if question.strip():
        with st.spinner('Searching and generating answer...'):
            try:
                answer, sources = rag.answer_question(question, k=top_k)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(answer)
                
                # Display sources
                st.markdown("### 📚 Sources")
                st.info(f"**Referenced documents:** {', '.join(sources)}")
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with Streamlit • Data: Fictional Riverside Public Library
    </div>
    """,
    unsafe_allow_html=True
)
