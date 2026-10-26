"""
RAG (Retrieval-Augmented Generation) System

A modular system for document ingestion, embedding, retrieval, and question answering.
Supports both OpenAI API and fully offline fallback modes.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    """Represents a source document."""
    content: str
    source: str
    metadata: Dict = None


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    text: str
    source: str
    chunk_id: int
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunk: Chunk
    score: float


class DocumentLoader:
    """Loads documents from various file formats."""
    
    @staticmethod
    def load_text(filepath: Path) -> str:
        """Load plain text file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_markdown(filepath: Path) -> str:
        """Load markdown file (as plain text)."""
        return DocumentLoader.load_text(filepath)
    
    @staticmethod
    def load_documents(doc_dir: Path) -> List[Document]:
        """Load all supported documents from directory."""
        documents = []
        supported_extensions = {'.txt', '.md', '.markdown'}
        
        for filepath in doc_dir.rglob('*'):
            if filepath.suffix.lower() in supported_extensions:
                content = DocumentLoader.load_text(filepath)
                documents.append(Document(
                    content=content,
                    source=str(filepath.name),
                    metadata={'path': str(filepath)}
                ))
        
        return documents


class TextChunker:
    """Splits documents into chunks for embedding."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source: str) -> List[Chunk]:
        """Split text into overlapping chunks."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # At least 50% through
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    source=source,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
            
            start = end - self.overlap
        
        return chunks


class EmbeddingModel:
    """Abstract base for embedding models."""
    
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingModel):
    """Offline embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            # Use the new method name if available, fall back to old one
            if hasattr(self.model, 'get_embedding_dimension'):
                self.dim = self.model.get_embedding_dimension()
            else:
                self.dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI API embedding."""
    
    def __init__(self, api_key: str, model: str = 'text-embedding-3-small'):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            self.dim = 1536  # text-embedding-3-small dimension
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)


class VectorStore:
    """Vector storage and retrieval using FAISS."""
    
    def __init__(self, embedding_dim: int):
        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.chunks: List[Chunk] = []
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. "
                "Install with: pip install faiss-cpu"
            )
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks with embeddings to the index."""
        if not chunks:
            return
        
        embeddings = np.array([chunk.embedding for chunk in chunks])
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[RetrievalResult]:
        """Search for similar chunks."""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(dist)
                ))
        
        return results
    
    def save(self, path: Path):
        """Save index and chunks to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.faiss.write_index(self.index, str(path / 'index.faiss'))
        
        # Save chunks
        chunks_data = [
            {
                'text': chunk.text,
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'embedding': chunk.embedding.tolist()
            }
            for chunk in self.chunks
        ]
        with open(path / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)
    
    def load(self, path: Path):
        """Load index and chunks from disk."""
        # Load FAISS index
        self.index = self.faiss.read_index(str(path / 'index.faiss'))
        
        # Load chunks
        with open(path / 'chunks.json', 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = [
            Chunk(
                text=c['text'],
                source=c['source'],
                chunk_id=c['chunk_id'],
                embedding=np.array(c['embedding'])
            )
            for c in chunks_data
        ]


class AnswerGenerator:
    """Generates answers from retrieved chunks."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4o-mini'):
        self.api_key = api_key
        self.model = model
        
        if api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                self.mode = 'openai'
            except ImportError:
                print("Warning: OpenAI not available, using extractive mode")
                self.mode = 'extractive'
        else:
            self.mode = 'extractive'
    
    def generate_answer(
        self, 
        question: str, 
        results: List[RetrievalResult]
    ) -> Tuple[str, List[str]]:
        """Generate answer from question and retrieved chunks."""
        if self.mode == 'openai':
            return self._generate_with_openai(question, results)
        else:
            return self._generate_extractive(question, results)
    
    def _generate_with_openai(
        self, 
        question: str, 
        results: List[RetrievalResult]
    ) -> Tuple[str, List[str]]:
        """Generate answer using OpenAI API."""
        context = "\n\n".join([
            f"[Source: {r.chunk.source}]\n{r.chunk.text}"
            for r in results
        ])
        
        prompt = f"""Answer the question based on the context below. 
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        answer = response.choices[0].message.content
        sources = list(set([r.chunk.source for r in results]))
        
        return answer, sources
    
    def _generate_extractive(
        self, 
        question: str, 
        results: List[RetrievalResult]
    ) -> Tuple[str, List[str]]:
        """Generate answer using simple extractive approach (offline fallback)."""
        if not results:
            return "I don't have enough information to answer that.", []
        
        # Use the most relevant chunk as the answer
        best_chunk = results[0].chunk
        answer = best_chunk.text
        
        # Trim if too long
        if len(answer) > 400:
            sentences = answer.split('.')
            answer = '. '.join(sentences[:3]) + '.'
        
        sources = list(set([r.chunk.source for r in results]))
        
        return answer, sources


class RAGSystem:
    """Main RAG system orchestrating all components."""
    
    def __init__(
        self, 
        embedding_model: EmbeddingModel,
        answer_generator: AnswerGenerator,
        vector_store: Optional[VectorStore] = None
    ):
        self.embedding_model = embedding_model
        self.answer_generator = answer_generator
        self.vector_store = vector_store or VectorStore(embedding_model.dim)
        self.chunker = TextChunker()
    
    def ingest_documents(self, documents: List[Document]):
        """Process and index documents."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunker.chunk_text(doc.content, doc.source)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_model.embed(texts)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to vector store
        self.vector_store.add_chunks(all_chunks)
        
        return len(all_chunks)
    
    def answer_question(
        self, 
        question: str, 
        k: int = 3
    ) -> Tuple[str, List[str]]:
        """Answer a question using RAG."""
        # Embed question
        query_embedding = self.embedding_model.embed([question])[0]
        
        # Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, k=k)
        
        # Generate answer
        answer, sources = self.answer_generator.generate_answer(question, results)
        
        return answer, sources
    
    def save(self, path: Path):
        """Save the RAG system state."""
        self.vector_store.save(path)
    
    def load(self, path: Path):
        """Load the RAG system state."""
        self.vector_store.load(path)


def build_rag_system(
    docs_dir: Path = Path('sample_docs'),
    index_dir: Path = Path('vector_index'),
    api_key: Optional[str] = None,
    force_rebuild: bool = False
) -> RAGSystem:
    """Build or load a RAG system."""
    
    # Choose embedding model
    if api_key:
        print("Using OpenAI embeddings and generation")
        embedding_model = OpenAIEmbedding(api_key)
    else:
        print("Using offline mode (sentence-transformers)")
        embedding_model = SentenceTransformerEmbedding()
    
    # Initialize answer generator
    answer_generator = AnswerGenerator(api_key)
    
    # Initialize RAG system
    rag = RAGSystem(embedding_model, answer_generator)
    
    # Load or build index
    if index_dir.exists() and not force_rebuild:
        print(f"Loading existing index from {index_dir}")
        rag.load(index_dir)
    else:
        print(f"Building new index from {docs_dir}")
        documents = DocumentLoader.load_documents(docs_dir)
        print(f"Loaded {len(documents)} documents")
        
        num_chunks = rag.ingest_documents(documents)
        print(f"Created {num_chunks} chunks")
        
        rag.save(index_dir)
        print(f"Saved index to {index_dir}")
    
    return rag
