#!/usr/bin/env python3
"""
Command-line interface for the RAG Q&A system.
"""
import argparse
from rag_system import RAGSystem


def main():
    parser = argparse.ArgumentParser(
        description="RAG Q&A System - Query documents using retrieval-augmented generation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents and build index')
    ingest_parser.add_argument(
        '--docs-dir',
        default='sample_docs',
        help='Directory containing documents (default: sample_docs)'
    )
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the document collection')
    query_parser.add_argument(
        'question',
        nargs='+',
        help='Question to ask'
    )
    query_parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of chunks to retrieve (default: 5)'
    )
    query_parser.add_argument(
        '--docs-dir',
        default='sample_docs',
        help='Directory containing documents (default: sample_docs)'
    )
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive query mode')
    interactive_parser.add_argument(
        '--docs-dir',
        default='sample_docs',
        help='Directory containing documents (default: sample_docs)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        print("=" * 60)
        print("RAG SYSTEM - DOCUMENT INGESTION")
        print("=" * 60)
        rag = RAGSystem(docs_dir=args.docs_dir)
        rag.ingest_documents()
    
    elif args.command == 'query':
        question = ' '.join(args.question)
        print("=" * 60)
        print("RAG SYSTEM - QUERY")
        print("=" * 60)
        print(f"\nQuestion: {question}\n")
        
        rag = RAGSystem(docs_dir=args.docs_dir)
        result = rag.query(question, k=args.k)
        
        print("Answer:")
        print("-" * 60)
        print(result['answer'])
        print("\n" + "=" * 60)
        print("SOURCES:")
        print("=" * 60)
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. Document: {source['doc_id']}")
            print(f"   Score: {source['score']:.4f}")
            print(f"   Text: {source['text']}")
    
    elif args.command == 'interactive':
        print("=" * 60)
        print("RAG SYSTEM - INTERACTIVE MODE")
        print("=" * 60)
        print("\nLoading system...")
        
        rag = RAGSystem(docs_dir=args.docs_dir)
        rag.load_index()
        
        print("\nReady! Type your questions (or 'quit' to exit)\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                print("\nProcessing...\n")
                result = rag.query(question)
                
                print("Answer:")
                print("-" * 60)
                print(result['answer'])
                print("\n" + "-" * 60 + "\n")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
