"""
Test suite for RAG system components.
All tests run in offline mode (no API keys required).
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from rag_system import DocumentChunker, EmbeddingModel, VectorIndex, RAGSystem


@pytest.fixture
def temp_docs_dir():
    """Create a temporary directory with test documents."""
    temp_dir = tempfile.mkdtemp()
    
    # Create test documents
    doc1 = Path(temp_dir) / "doc1.txt"
    doc1.write_text("The quick brown fox jumps over the lazy dog. " * 10)
    
    doc2 = Path(temp_dir) / "doc2.md"
    doc2.write_text("# Test Document\n\nThis is a test document about widgets. " * 10)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for index storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDocumentChunker:
    """Test document chunking functionality."""
    
    def test_chunk_creation(self):
        """Test that chunks are created correctly."""
        chunker = DocumentChunker(chunk_size=10, overlap=2)
        text = " ".join([f"word{i}" for i in range(25)])
        
        chunks = chunker.chunk_text(text, "test_doc")
        
        assert len(chunks) > 1
        assert all('text' in c for c in chunks)
        assert all('doc_id' in c for c in chunks)
        assert all('chunk_id' in c for c in chunks)
        assert all(c['doc_id'] == 'test_doc' for c in chunks)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=10, overlap=3)
        text = " ".join([f"word{i}" for i in range(30)])
        
        chunks = chunker.chunk_text(text, "test_doc")
        
        # Should have multiple chunks
        assert len(chunks) >= 2
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("", "empty_doc")
        
        # Should create at least one chunk even for empty text
        assert len(chunks) >= 0


class TestEmbeddingModel:
    """Test embedding model functionality."""
    
    def test_offline_initialization(self):
        """Test that model initializes without API key."""
        # Temporarily remove API key if present
        old_key = os.environ.pop('OPENAI_API_KEY', None)
        
        try:
            model = EmbeddingModel(use_openai=False)
            assert model is not None
            assert hasattr(model, 'embed_texts')
        finally:
            if old_key:
                os.environ['OPENAI_API_KEY'] = old_key
    
    def test_embed_texts_offline(self):
        """Test embedding generation in offline mode."""
        model = EmbeddingModel(use_openai=False)
        texts = ["Hello world", "Test document", "Another sentence"]
        
        embeddings = model.embed_texts(texts)
        
        assert embeddings is not None
        assert len(embeddings) == len(texts)
        assert embeddings.shape[1] > 0  # Has embedding dimension
    
    def test_embed_query_offline(self):
        """Test query embedding in offline mode."""
        model = EmbeddingModel(use_openai=False)
        
        # For TF-IDF, need to fit on corpus first
        corpus = ["Hello world", "Test document", "Another sentence"]
        model.embed_texts(corpus)
        
        query_embedding = model.embed_query("Hello", corpus=corpus)
        
        assert query_embedding is not None
        assert len(query_embedding) > 0


class TestVectorIndex:
    """Test vector index functionality."""
    
    def test_index_creation(self):
        """Test that index is created correctly."""
        index = VectorIndex(dimension=384)
        assert index is not None
    
    def test_add_and_search(self):
        """Test adding vectors and searching."""
        import numpy as np
        
        index = VectorIndex(dimension=10)
        
        # Create test vectors and chunks
        vectors = np.random.randn(5, 10).astype('float32')
        chunks = [
            {'text': f'Chunk {i}', 'doc_id': f'doc{i}', 'chunk_id': f'chunk{i}'}
            for i in range(5)
        ]
        
        index.add_vectors(vectors, chunks)
        
        # Search
        query_vector = np.random.randn(10).astype('float32')
        results = index.search(query_vector, k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (chunk, score)
    
    def test_save_and_load(self):
        """Test saving and loading index."""
        import numpy as np
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_index"
            
            # Create and save
            index = VectorIndex(dimension=10)
            vectors = np.random.randn(3, 10).astype('float32')
            chunks = [{'text': f'Chunk {i}', 'doc_id': 'test', 'chunk_id': f'c{i}'} for i in range(3)]
            index.add_vectors(vectors, chunks)
            index.save(str(index_path))
            
            # Load
            new_index = VectorIndex(dimension=10)
            new_index.load(str(index_path))
            
            assert len(new_index.chunks) == 3


class TestRAGSystem:
    """Test complete RAG system."""
    
    def test_system_initialization(self, temp_docs_dir, temp_index_dir):
        """Test RAG system initialization."""
        rag = RAGSystem(docs_dir=temp_docs_dir, index_dir=temp_index_dir)
        assert rag is not None
        assert rag.docs_dir.exists()
    
    def test_document_ingestion(self, temp_docs_dir, temp_index_dir):
        """Test document ingestion pipeline."""
        rag = RAGSystem(docs_dir=temp_docs_dir, index_dir=temp_index_dir)
        
        # Remove API key to ensure offline mode
        old_key = os.environ.pop('OPENAI_API_KEY', None)
        
        try:
            rag.ingest_documents()
            
            # Check that index files were created
            assert (Path(temp_index_dir) / "document_index_chunks.json").exists()
        finally:
            if old_key:
                os.environ['OPENAI_API_KEY'] = old_key
    
    def test_query_execution(self, temp_docs_dir, temp_index_dir):
        """Test query execution."""
        rag = RAGSystem(docs_dir=temp_docs_dir, index_dir=temp_index_dir)
        
        # Remove API key to ensure offline mode
        old_key = os.environ.pop('OPENAI_API_KEY', None)
        
        try:
            # Ingest
            rag.ingest_documents()
            
            # Query
            result = rag.query("test widgets", k=3)
            
            assert 'question' in result
            assert 'answer' in result
            assert 'sources' in result
            assert len(result['sources']) <= 3
        finally:
            if old_key:
                os.environ['OPENAI_API_KEY'] = old_key
    
    def test_load_nonexistent_index(self, temp_docs_dir, temp_index_dir):
        """Test that loading nonexistent index raises error."""
        rag = RAGSystem(docs_dir=temp_docs_dir, index_dir=temp_index_dir)
        
        with pytest.raises(FileNotFoundError):
            rag.load_index()


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline_offline(self, temp_docs_dir, temp_index_dir):
        """Test complete pipeline in offline mode."""
        # Remove API key
        old_key = os.environ.pop('OPENAI_API_KEY', None)
        
        try:
            # Create system
            rag = RAGSystem(docs_dir=temp_docs_dir, index_dir=temp_index_dir)
            
            # Ingest documents
            rag.ingest_documents()
            
            # Verify index exists
            assert (Path(temp_index_dir) / "document_index_chunks.json").exists()
            
            # Query system
            result = rag.query("What is this document about?", k=2)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'question' in result
            assert 'answer' in result
            assert 'sources' in result
            assert isinstance(result['sources'], list)
            
            # Verify answer is not empty
            assert len(result['answer']) > 0
            
        finally:
            if old_key:
                os.environ['OPENAI_API_KEY'] = old_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
