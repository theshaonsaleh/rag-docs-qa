"""
Evaluation script for RAG system retrieval accuracy.
Tests if the system retrieves correct source documents for known questions.
"""
from typing import List, Dict
from rag_system import RAGSystem


# Test cases: questions with expected source documents
EVAL_CASES = [
    {
        "question": "What are the standard working hours?",
        "expected_sources": ["employee_handbook"]
    },
    {
        "question": "How much vacation leave do employees get?",
        "expected_sources": ["employee_handbook"]
    },
    {
        "question": "What is the price of a Series A compact widget?",
        "expected_sources": ["product_catalog"]
    },
    {
        "question": "What materials are Series B widgets made from?",
        "expected_sources": ["product_catalog", "technical_specs"]
    },
    {
        "question": "What is the company's parking policy?",
        "expected_sources": ["company_policies"]
    },
    {
        "question": "What PPE is required in manufacturing areas?",
        "expected_sources": ["safety_guidelines"]
    },
    {
        "question": "What is the temperature range for Series B widgets?",
        "expected_sources": ["technical_specs", "product_catalog"]
    },
    {
        "question": "How often are performance reviews conducted?",
        "expected_sources": ["employee_handbook"]
    },
    {
        "question": "What is the expense reimbursement policy?",
        "expected_sources": ["company_policies"]
    },
    {
        "question": "What standards do widgets comply with?",
        "expected_sources": ["technical_specs"]
    }
]


def evaluate_retrieval(rag_system: RAGSystem, test_cases: List[Dict], k: int = 5) -> Dict:
    """
    Evaluate retrieval accuracy.
    
    Returns metrics:
    - hit_rate: percentage of queries where at least one expected source was retrieved
    - mrr: mean reciprocal rank of first correct source
    - precision@k: average precision at k
    """
    hits = 0
    reciprocal_ranks = []
    precisions = []
    
    results = []
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    for i, case in enumerate(test_cases, 1):
        question = case['question']
        expected = set(case['expected_sources'])
        
        # Query the system
        result = rag_system.query(question, k=k)
        retrieved_sources = [s['doc_id'] for s in result['sources']]
        
        # Check if any expected source was retrieved
        retrieved_set = set(retrieved_sources)
        hit = bool(expected & retrieved_set)
        if hit:
            hits += 1
        
        # Calculate reciprocal rank
        rr = 0
        for rank, source in enumerate(retrieved_sources, 1):
            if source in expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        
        # Calculate precision@k
        relevant_retrieved = len(expected & retrieved_set)
        precision = relevant_retrieved / k
        precisions.append(precision)
        
        # Store detailed result
        results.append({
            'question': question,
            'expected': list(expected),
            'retrieved': retrieved_sources[:3],  # Top 3
            'hit': hit,
            'rr': rr,
            'precision': precision
        })
        
        # Print result
        status = "✓" if hit else "✗"
        print(f"\n{status} Test {i}: {question}")
        print(f"   Expected: {', '.join(expected)}")
        print(f"   Retrieved: {', '.join(retrieved_sources[:3])}")
        print(f"   Hit: {hit}, RR: {rr:.3f}, P@{k}: {precision:.3f}")
    
    # Calculate overall metrics
    hit_rate = hits / len(test_cases)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_precision = sum(precisions) / len(precisions)
    
    print("\n" + "=" * 70)
    print("SUMMARY METRICS")
    print("=" * 70)
    print(f"Hit Rate@{k}:        {hit_rate:.1%} ({hits}/{len(test_cases)} queries)")
    print(f"Mean Reciprocal Rank: {mrr:.3f}")
    print(f"Precision@{k}:        {avg_precision:.3f}")
    print("=" * 70)
    
    return {
        'hit_rate': hit_rate,
        'mrr': mrr,
        'precision_at_k': avg_precision,
        'total_cases': len(test_cases),
        'hits': hits,
        'results': results
    }


def main():
    print("Loading RAG system...")
    rag = RAGSystem()
    
    try:
        rag.load_index()
    except FileNotFoundError:
        print("\nError: Index not found. Please run ingestion first:")
        print("  python cli.py ingest")
        return
    
    print("\nRunning evaluation with {} test cases...\n".format(len(EVAL_CASES)))
    
    metrics = evaluate_retrieval(rag, EVAL_CASES, k=5)
    
    print("\nEvaluation complete!")
    
    # Interpretation
    print("\nInterpretation:")
    if metrics['hit_rate'] >= 0.9:
        print("✓ Excellent retrieval performance!")
    elif metrics['hit_rate'] >= 0.7:
        print("✓ Good retrieval performance")
    elif metrics['hit_rate'] >= 0.5:
        print("⚠ Moderate retrieval performance - consider tuning chunk size or k")
    else:
        print("✗ Poor retrieval performance - review chunking and embedding strategy")


if __name__ == "__main__":
    main()
