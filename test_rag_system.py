"""
Unit tests for the RAG system components.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from rag_system import (
    Document, Chunk, DocumentLoader, TextChunker,
    SentenceTransformerEmbedding, VectorStore
)


class TestDocumentLoader(unittest.TestCase):
    """Test document loading functionality."""
    
    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_load_text_file(self):
        """Test loading a plain text file."""
        test_file = self.test_dir / "test.txt"
        content = "This is a test document."
        test_file.write_text(content)
        
        loaded = DocumentLoader.load_text(test_file)
        self.assertEqual(loaded, content)
    
    def test_load_markdown_file(self):
        """Test loading a markdown file."""
        test_file = self.test_dir / "test.md"
        content = "# Test\n\nThis is markdown content."
        test_file.write_text(content)
        
        loaded = DocumentLoader.load_markdown(test_file)
        self.assertEqual(loaded, content)
    
    def test_load_documents(self):
        """Test loading multiple documents from directory."""
        (self.test_dir / "doc1.txt").write_text("Document 1")
        (self.test_dir / "doc2.md").write_text("Document 2")
        (self.test_dir / "ignore.pdf").write_text("Should be ignored")
        
        docs = DocumentLoader.load_documents(self.test_dir)
        
        self.assertEqual(len(docs), 2)
        sources = [d.source for d in docs]
        self.assertIn("doc1.txt", sources)
        self.assertIn("doc2.md", sources)


class TestTextChunker(unittest.TestCase):
    """Test text chunking functionality."""
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "This is a short text."
        chunks = chunker.chunk_text(text, "test.txt")
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].source, "test.txt")
    
    def test_chunk_long_text(self):
        """Test chunking long text."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "A " * 100  # 200 characters
        chunks = chunker.chunk_text(text, "test.txt")
        
        self.assertGreater(len(chunks), 1)
        
        # Check chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, i)
    
    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "Word " * 50
        chunks = chunker.chunk_text(text, "test.txt")
        
        if len(chunks) > 1:
            # Check that some content overlaps
            self.assertTrue(any(
                word in chunks[1].text 
                for word in chunks[0].text.split()[-3:]
            ))


class TestSentenceTransformerEmbedding(unittest.TestCase):
    """Test sentence transformer embeddings."""
    
    def setUp(self):
        """Initialize embedding model."""
        try:
            self.model = SentenceTransformerEmbedding()
        except ImportError:
            self.skipTest("sentence-transformers not installed")
    
    def test_embed_single_text(self):
        """Test embedding a single text."""
        texts = ["This is a test sentence."]
        embeddings = self.model.embed(texts)
        
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], self.model.dim)
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = self.model.embed(texts)
        
        self.assertEqual(embeddings.shape[0], 3)
        self.assertEqual(embeddings.shape[1], self.model.dim)
    
    def test_similar_texts_have_similar_embeddings(self):
        """Test that similar texts have similar embeddings."""
        texts = [
            "The cat sat on the mat.",
            "A cat was sitting on a mat.",
            "The dog ran in the park."
        ]
        embeddings = self.model.embed(texts)
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_cat = cosine_similarity(embeddings[0], embeddings[1])
        sim_dog = cosine_similarity(embeddings[0], embeddings[2])
        
        # Cat sentences should be more similar to each other
        self.assertGreater(sim_cat, sim_dog)


class TestVectorStore(unittest.TestCase):
    """Test vector store functionality."""
    
    def setUp(self):
        """Set up test chunks and vector store."""
        try:
            self.store = VectorStore(embedding_dim=384)
            
            # Create test chunks with embeddings
            self.chunks = [
                Chunk(
                    text="Library hours are Monday to Friday 9am to 5pm.",
                    source="hours.txt",
                    chunk_id=0,
                    embedding=np.random.rand(384)
                ),
                Chunk(
                    text="The library offers free WiFi to all visitors.",
                    source="services.txt",
                    chunk_id=1,
                    embedding=np.random.rand(384)
                ),
                Chunk(
                    text="Children's programs include story time on Tuesdays.",
                    source="programs.txt",
                    chunk_id=2,
                    embedding=np.random.rand(384)
                )
            ]
        except ImportError:
            self.skipTest("faiss-cpu not installed")
    
    def test_add_chunks(self):
        """Test adding chunks to vector store."""
        self.store.add_chunks(self.chunks)
        self.assertEqual(len(self.store.chunks), 3)
    
    def test_search(self):
        """Test searching for similar chunks."""
        self.store.add_chunks(self.chunks)
        
        query_embedding = np.random.rand(384)
        results = self.store.search(query_embedding, k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0].chunk)
        self.assertIsInstance(results[0].score, float)
    
    def test_save_and_load(self):
        """Test saving and loading vector store."""
        self.store.add_chunks(self.chunks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            self.store.save(save_path)
            
            # Load into new store
            new_store = VectorStore(embedding_dim=384)
            new_store.load(save_path)
            
            self.assertEqual(len(new_store.chunks), len(self.chunks))
            self.assertEqual(new_store.chunks[0].text, self.chunks[0].text)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete RAG pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from sentence_transformers import SentenceTransformer
            self.has_deps = True
        except ImportError:
            self.has_deps = False
    
    def test_end_to_end_pipeline(self):
        """Test the complete RAG pipeline."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")
        
        from rag_system import RAGSystem, SentenceTransformerEmbedding, AnswerGenerator
        
        # Create test documents
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            
            (docs_dir / "test1.txt").write_text(
                "The library is open Monday to Friday from 9 AM to 5 PM."
            )
            (docs_dir / "test2.txt").write_text(
                "The library offers free WiFi to all visitors."
            )
            
            # Build RAG system
            embedding_model = SentenceTransformerEmbedding()
            answer_generator = AnswerGenerator(api_key=None)
            rag = RAGSystem(embedding_model, answer_generator)
            
            # Load and process documents
            documents = DocumentLoader.load_documents(docs_dir)
            num_chunks = rag.ingest_documents(documents)
            
            self.assertGreater(num_chunks, 0)
            
            # Ask a question
            question = "What are the library hours?"
            answer, sources = rag.answer_question(question, k=2)
            
            self.assertIsNotNone(answer)
            self.assertGreater(len(sources), 0)


if __name__ == '__main__':
    unittest.main()
