"""
Evaluation script for the RAG system.

Measures retrieval hit-rate: what percentage of questions retrieve at least one relevant chunk.
"""

import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

from rag_system import build_rag_system


# Test Q&A pairs with expected source documents
EVAL_CASES = [
    {
        "question": "What are the library hours on weekdays?",
        "expected_sources": ["library_hours.md"],
        "answer_keywords": ["Monday", "Thursday", "9:00", "8:00"]
    },
    {
        "question": "How much does it cost to print in color?",
        "expected_sources": ["computer_policy.txt"],
        "answer_keywords": ["$0.50", "color"]
    },
    {
        "question": "Can non-residents get a library card?",
        "expected_sources": ["membership_policy.md"],
        "answer_keywords": ["$50", "annual", "non-resident"]
    },
    {
        "question": "What is the maximum capacity of Community Room A?",
        "expected_sources": ["meeting_rooms.md"],
        "answer_keywords": ["50", "people"]
    },
    {
        "question": "What digital platforms does the library offer for e-books?",
        "expected_sources": ["digital_resources.md"],
        "answer_keywords": ["OverDrive", "Libby", "Hoopla"]
    },
    {
        "question": "When is preschool story time?",
        "expected_sources": ["childrens_programs.md"],
        "answer_keywords": ["Tuesday", "Thursday", "10:30"]
    },
    {
        "question": "Is WiFi available at the library?",
        "expected_sources": ["computer_policy.txt"],
        "answer_keywords": ["WiFi", "free", "Riverside_Library_WiFi"]
    },
    {
        "question": "How long can I borrow a DVD?",
        "expected_sources": ["membership_policy.md"],
        "answer_keywords": ["1 week", "DVD"]
    },
    {
        "question": "What is the refundable deposit for meeting rooms?",
        "expected_sources": ["meeting_rooms.md"],
        "answer_keywords": ["$50", "deposit"]
    },
    {
        "question": "What homework help programs are available?",
        "expected_sources": ["childrens_programs.md"],
        "answer_keywords": ["After-School", "Monday-Thursday", "3:30"]
    }
]


def evaluate_retrieval(rag, test_cases: List[dict], k: int = 3) -> dict:
    """Evaluate retrieval performance."""
    results = []
    hits = 0
    total = len(test_cases)
    
    print("Running evaluation...\n")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected_sources = case["expected_sources"]
        
        # Get embeddings and retrieve
        query_embedding = rag.embedding_model.embed([question])[0]
        retrieved = rag.vector_store.search(query_embedding, k=k)
        
        # Check if any expected source was retrieved
        retrieved_sources = [r.chunk.source for r in retrieved]
        hit = any(src in retrieved_sources for src in expected_sources)
        
        if hit:
            hits += 1
        
        results.append({
            "question": question,
            "expected": expected_sources,
            "retrieved": retrieved_sources,
            "hit": hit
        })
        
        # Print result
        status = "✓ HIT" if hit else "✗ MISS"
        print(f"{i}. {status}")
        print(f"   Q: {question}")
        print(f"   Expected: {', '.join(expected_sources)}")
        print(f"   Retrieved: {', '.join(retrieved_sources)}")
        print()
    
    print("=" * 80)
    
    hit_rate = hits / total
    
    return {
        "total": total,
        "hits": hits,
        "misses": total - hits,
        "hit_rate": hit_rate,
        "results": results
    }


def evaluate_answer_quality(rag, test_cases: List[dict], k: int = 3) -> dict:
    """Evaluate answer quality by checking if keywords are present."""
    results = []
    keyword_matches = 0
    total = len(test_cases)
    
    print("\nEvaluating answer quality...\n")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        keywords = case["answer_keywords"]
        
        # Generate answer
        answer, sources = rag.answer_question(question, k=k)
        
        # Check if answer contains expected keywords
        answer_lower = answer.lower()
        matched_keywords = [kw for kw in keywords if kw.lower() in answer_lower]
        has_match = len(matched_keywords) > 0
        
        if has_match:
            keyword_matches += 1
        
        results.append({
            "question": question,
            "answer": answer,
            "keywords": keywords,
            "matched": matched_keywords,
            "has_match": has_match
        })
        
        # Print result
        status = "✓ PASS" if has_match else "✗ FAIL"
        print(f"{i}. {status}")
        print(f"   Q: {question}")
        print(f"   Keywords: {', '.join(keywords)}")
        print(f"   Matched: {', '.join(matched_keywords) if matched_keywords else 'None'}")
        print(f"   A: {answer[:100]}...")
        print()
    
    print("=" * 80)
    
    keyword_score = keyword_matches / total
    
    return {
        "total": total,
        "keyword_matches": keyword_matches,
        "keyword_score": keyword_score,
        "results": results
    }


def main():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("=" * 80)
    print("RAG SYSTEM EVALUATION")
    print("=" * 80)
    print()
    
    # Build RAG system
    mode = "OpenAI" if api_key else "Offline"
    print(f"Mode: {mode}")
    print()
    
    rag = build_rag_system(
        docs_dir=Path('sample_docs'),
        index_dir=Path('vector_index'),
        api_key=api_key,
        force_rebuild=False
    )
    print()
    
    # Run retrieval evaluation
    retrieval_metrics = evaluate_retrieval(rag, EVAL_CASES, k=3)
    
    # Run answer quality evaluation
    answer_metrics = evaluate_answer_quality(rag, EVAL_CASES, k=3)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()
    print(f"Mode: {mode}")
    print()
    print("Retrieval Performance:")
    print(f"  Total questions: {retrieval_metrics['total']}")
    print(f"  Hits: {retrieval_metrics['hits']}")
    print(f"  Misses: {retrieval_metrics['misses']}")
    print(f"  Hit Rate: {retrieval_metrics['hit_rate']:.1%}")
    print()
    print("Answer Quality:")
    print(f"  Total questions: {answer_metrics['total']}")
    print(f"  Keyword matches: {answer_metrics['keyword_matches']}")
    print(f"  Keyword score: {answer_metrics['keyword_score']:.1%}")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
