"""
Streamlit web UI for the RAG Q&A system.
"""
import streamlit as st
from pathlib import Path
from rag_system import RAGSystem


# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 RAG Document Q&A System")
st.markdown("""
Ask questions about the document collection. The system retrieves relevant passages 
and generates answers using retrieval-augmented generation (RAG).
""")

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    docs_dir = st.text_input("Documents Directory", value="sample_docs")
    k_value = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    
    # System status
    st.header("📊 System Status")
    
    if Path(docs_dir).exists():
        doc_count = len(list(Path(docs_dir).glob("*.md"))) + \
                   len(list(Path(docs_dir).glob("*.txt"))) + \
                   len(list(Path(docs_dir).glob("*.pdf")))
        st.success(f"✓ Found {doc_count} documents")
    else:
        st.error(f"✗ Directory '{docs_dir}' not found")
    
    index_exists = Path("indexes/document_index_chunks.json").exists()
    if index_exists:
        st.success("✓ Index loaded")
    else:
        st.warning("⚠ Index not found. Click 'Build Index' below.")
    
    st.markdown("---")
    
    # Build index button
    if st.button("🔨 Build Index", use_container_width=True):
        with st.spinner("Building index... This may take a minute."):
            try:
                rag = RAGSystem(docs_dir=docs_dir)
                rag.ingest_documents()
                st.session_state.rag_system = None
                st.session_state.initialized = False
                st.success("Index built successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error building index: {e}")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This is a learning project demonstrating RAG (Retrieval-Augmented Generation).
    
    **Features:**
    - Document chunking & embedding
    - Vector similarity search
    - LLM-powered answers with source citations
    - Offline mode support
    """)

# Main content area
if not index_exists:
    st.warning("⚠️ Index not found. Please build the index using the sidebar button.")
else:
    # Initialize RAG system
    if not st.session_state.initialized:
        with st.spinner("Loading RAG system..."):
            try:
                st.session_state.rag_system = RAGSystem(docs_dir=docs_dir)
                st.session_state.rag_system.load_index()
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"Error loading system: {e}")
                st.stop()
    
    # Query interface
    st.header("💬 Ask a Question")
    
    # Sample questions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📅 What are the working hours?"):
            st.session_state.current_query = "What are the working hours?"
    with col2:
        if st.button("💰 What is the vacation policy?"):
            st.session_state.current_query = "What is the vacation policy?"
    with col3:
        if st.button("🔧 Tell me about Series B widgets"):
            st.session_state.current_query = "Tell me about Series B widgets"
    
    # Question input
    question = st.text_input(
        "Your Question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., What are the safety requirements in manufacturing areas?",
        key="question_input"
    )
    
    col_ask, col_clear = st.columns([1, 4])
    with col_ask:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Process query
    if ask_button and question:
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = st.session_state.rag_system.query(question, k=k_value)
                st.session_state.history.insert(0, result)
                st.session_state.current_query = ''
                st.rerun()
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    # Display results
    if st.session_state.history:
        st.markdown("---")
        st.header("📝 Results")
        
        for i, result in enumerate(st.session_state.history):
            with st.expander(f"Q: {result['question']}", expanded=(i == 0)):
                st.markdown("### Answer")
                st.markdown(result['answer'])
                
                st.markdown("### 📚 Sources")
                for j, source in enumerate(result['sources'], 1):
                    with st.container():
                        st.markdown(f"**{j}. {source['doc_id']}** (score: {source['score']:.4f})")
                        st.text(source['text'])
                        st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    RAG Document Q&A System | Personal, non-commercial learning project
    </div>
    """,
    unsafe_allow_html=True
)
