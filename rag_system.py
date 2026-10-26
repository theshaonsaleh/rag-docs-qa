"""
Core RAG (Retrieval-Augmented Generation) system.
Supports both online (OpenAI) and offline (sentence-transformers/TF-IDF) modes.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class DocumentChunker:
    """Split documents into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{len(chunks)}"
            })
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks


class EmbeddingModel:
    """Wrapper for embedding models with online/offline fallback."""
    
    def __init__(self, use_openai: bool = None):
        """
        Initialize embedding model.
        
        Args:
            use_openai: If True, use OpenAI. If False, use sentence-transformers.
                       If None, auto-detect based on OPENAI_API_KEY.
        """
        if use_openai is None:
            use_openai = bool(os.getenv('OPENAI_API_KEY'))
        
        self.use_openai = use_openai
        
        if self.use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.model_name = "text-embedding-3-small"
                print(f"Using OpenAI embeddings: {self.model_name}")
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
                print("Falling back to sentence-transformers")
                self.use_openai = False
        
        if not self.use_openai:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_name = "all-MiniLM-L6-v2"
                print(f"Using sentence-transformers: {self.model_name}")
            except Exception as e:
                print(f"Sentence-transformers failed: {e}")
                print("Falling back to TF-IDF")
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.model = TfidfVectorizer(max_features=384)
                self.model_name = "TF-IDF"
                self.use_tfidf = True
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.use_openai:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        elif hasattr(self, 'use_tfidf'):
            # TF-IDF requires fitting on corpus first
            return self.model.fit_transform(texts).toarray()
        else:
            # Sentence-transformers
            return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_query(self, query: str, corpus: List[str] = None) -> np.ndarray:
        """Generate embedding for a single query."""
        if self.use_openai:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name
            )
            return np.array(response.data[0].embedding)
        elif hasattr(self, 'use_tfidf'):
            # TF-IDF needs the corpus to be fitted first
            if corpus is None:
                raise ValueError("TF-IDF requires corpus for query embedding")
            return self.model.transform([query]).toarray()[0]
        else:
            return self.model.encode([query], convert_to_numpy=True)[0]


class VectorIndex:
    """FAISS-based vector index for similarity search."""
    
    def __init__(self, dimension: int):
        """Initialize FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(dimension)
            self.use_faiss = True
        except ImportError:
            print("FAISS not available, using numpy fallback")
            self.vectors = None
            self.use_faiss = False
            self.dimension = dimension
        
        self.chunks = []
    
    def add_vectors(self, vectors: np.ndarray, chunks: List[Dict]):
        """Add vectors and corresponding chunks to the index."""
        if self.use_faiss:
            self.index.add(vectors.astype('float32'))
        else:
            if self.vectors is None:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])
        
        self.chunks.extend(chunks)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for top-k most similar chunks."""
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        if self.use_faiss:
            distances, indices = self.index.search(query_vector, k)
            results = [
                (self.chunks[idx], float(dist))
                for dist, idx in zip(distances[0], indices[0])
                if idx < len(self.chunks)
            ]
        else:
            # Numpy fallback: compute cosine similarity
            if self.vectors is None or len(self.vectors) == 0:
                return []
            
            # Normalize vectors
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
            
            # Compute similarities
            similarities = np.dot(vectors_norm, query_norm.T).flatten()
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            results = [
                (self.chunks[idx], float(1 - similarities[idx]))  # Convert similarity to distance
                for idx in top_k_indices
            ]
        
        return results
    
    def save(self, path: str):
        """Save index and chunks to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.use_faiss:
            import faiss
            faiss.write_index(self.index, f"{path}.faiss")
        else:
            np.save(f"{path}_vectors.npy", self.vectors)
        
        with open(f"{path}_chunks.json", 'w') as f:
            json.dump(self.chunks, f, indent=2)
    
    def load(self, path: str):
        """Load index and chunks from disk."""
        if self.use_faiss:
            import faiss
            self.index = faiss.read_index(f"{path}.faiss")
        else:
            self.vectors = np.load(f"{path}_vectors.npy")
        
        with open(f"{path}_chunks.json", 'r') as f:
            self.chunks = json.load(f)


class LLMAnswerer:
    """Generate answers using LLM or extractive fallback."""
    
    def __init__(self, use_openai: bool = None):
        """Initialize LLM client or fallback to extractive mode."""
        if use_openai is None:
            use_openai = bool(os.getenv('OPENAI_API_KEY'))
        
        self.use_openai = use_openai
        
        if self.use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
                print(f"Using OpenAI LLM: {self.model}")
            except Exception as e:
                print(f"OpenAI LLM initialization failed: {e}")
                print("Falling back to extractive answers")
                self.use_openai = False
        
        if not self.use_openai:
            print("Using extractive answer mode (no LLM)")
    
    def generate_answer(self, query: str, contexts: List[Tuple[Dict, float]]) -> str:
        """Generate answer from query and retrieved contexts."""
        if not contexts:
            return "I couldn't find any relevant information to answer your question."
        
        if self.use_openai:
            return self._generate_llm_answer(query, contexts)
        else:
            return self._generate_extractive_answer(query, contexts)
    
    def _generate_llm_answer(self, query: str, contexts: List[Tuple[Dict, float]]) -> str:
        """Generate answer using LLM."""
        context_text = "\n\n".join([
            f"[Source: {chunk['doc_id']}]\n{chunk['text']}"
            for chunk, _ in contexts
        ])
        
        prompt = f"""Answer the following question based only on the provided context. 
If the answer cannot be found in the context, say so.
Include source document names in your answer.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents. Always cite your sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._generate_extractive_answer(query, contexts)
    
    def _generate_extractive_answer(self, query: str, contexts: List[Tuple[Dict, float]]) -> str:
        """Generate answer by returning top relevant passages."""
        answer_parts = ["Based on the documents, here are the most relevant passages:\n"]
        
        for i, (chunk, score) in enumerate(contexts[:3], 1):
            answer_parts.append(f"\n{i}. From '{chunk['doc_id']}':")
            answer_parts.append(f"   {chunk['text'][:300]}...")
            answer_parts.append(f"   (relevance score: {score:.3f})\n")
        
        return "\n".join(answer_parts)


class RAGSystem:
    """Main RAG system orchestrating all components."""
    
    def __init__(self, docs_dir: str = "sample_docs", index_dir: str = "indexes"):
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.chunker = DocumentChunker()
        self.embedding_model = None
        self.vector_index = None
        self.llm = None
        self.all_chunks = []
    
    def ingest_documents(self):
        """Load and process all documents from docs_dir."""
        print(f"\nIngesting documents from {self.docs_dir}...")
        
        documents = []
        for ext in ['*.md', '*.txt', '*.pdf']:
            for doc_path in self.docs_dir.glob(ext):
                print(f"  Loading {doc_path.name}...")
                text = self._load_document(doc_path)
                if text:
                    documents.append((doc_path.stem, text))
        
        print(f"\nProcessed {len(documents)} documents")
        
        # Chunk documents
        print("Chunking documents...")
        all_chunks = []
        for doc_id, text in documents:
            chunks = self.chunker.chunk_text(text, doc_id)
            all_chunks.extend(chunks)
        
        self.all_chunks = all_chunks
        print(f"Created {len(all_chunks)} chunks")
        
        # Initialize embeddings
        print("\nInitializing embedding model...")
        self.embedding_model = EmbeddingModel()
        
        # Generate embeddings
        print("Generating embeddings...")
        chunk_texts = [c['text'] for c in all_chunks]
        embeddings = self.embedding_model.embed_texts(chunk_texts)
        
        # Build index
        print("Building vector index...")
        self.vector_index = VectorIndex(dimension=embeddings.shape[1])
        self.vector_index.add_vectors(embeddings, all_chunks)
        
        # Save index
        index_path = self.index_dir / "document_index"
        print(f"Saving index to {index_path}...")
        self.vector_index.save(str(index_path))
        
        print("\nIngestion complete!")
    
    def load_index(self):
        """Load pre-built index from disk."""
        index_path = self.index_dir / "document_index"
        
        if not Path(f"{index_path}_chunks.json").exists():
            raise FileNotFoundError(f"Index not found at {index_path}. Run ingestion first.")
        
        print("Loading index...")
        
        # Load chunks to get corpus for TF-IDF
        with open(f"{index_path}_chunks.json", 'r') as f:
            self.all_chunks = json.load(f)
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModel()
        
        # If using TF-IDF, fit on corpus
        if hasattr(self.embedding_model, 'use_tfidf'):
            chunk_texts = [c['text'] for c in self.all_chunks]
            self.embedding_model.embed_texts(chunk_texts)
        
        # Load vector index
        # Determine dimension from a sample embedding
        sample_text = self.all_chunks[0]['text'] if self.all_chunks else "sample"
        sample_embedding = self.embedding_model.embed_query(
            sample_text, 
            corpus=[c['text'] for c in self.all_chunks] if hasattr(self.embedding_model, 'use_tfidf') else None
        )
        
        self.vector_index = VectorIndex(dimension=len(sample_embedding))
        self.vector_index.load(str(index_path))
        
        # Initialize LLM
        self.llm = LLMAnswerer()
        
        print("Index loaded successfully!")
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Query the RAG system."""
        if self.vector_index is None:
            self.load_index()
        
        if self.llm is None:
            self.llm = LLMAnswerer()
        
        # Generate query embedding
        corpus = [c['text'] for c in self.all_chunks] if hasattr(self.embedding_model, 'use_tfidf') else None
        query_embedding = self.embedding_model.embed_query(question, corpus=corpus)
        
        # Retrieve relevant chunks
        results = self.vector_index.search(query_embedding, k=k)
        
        # Generate answer
        answer = self.llm.generate_answer(question, results)
        
        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'doc_id': chunk['doc_id'],
                    'text': chunk['text'][:200] + '...',
                    'score': score
                }
                for chunk, score in results
            ]
        }
    
    def _load_document(self, path: Path) -> Optional[str]:
        """Load document from file."""
        try:
            if path.suffix == '.pdf':
                return self._load_pdf(path)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
            return ""


if __name__ == "__main__":
    import sys
    
    rag = RAGSystem()
    
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        rag.ingest_documents()
    else:
        print("Usage: python rag_system.py ingest")
