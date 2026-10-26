#!/usr/bin/env python3
"""
Command-line interface for the RAG Q&A system.
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from rag_system import build_rag_system


def main():
    parser = argparse.ArgumentParser(
        description='RAG Q&A System - Ask questions about your documents'
    )
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=Path('sample_docs'),
        help='Directory containing documents to index'
    )
    parser.add_argument(
        '--index-dir',
        type=Path,
        default=Path('vector_index'),
        help='Directory to store/load vector index'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild of index even if it exists'
    )
    parser.add_argument(
        '--question',
        '-q',
        type=str,
        help='Question to ask (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of chunks to retrieve (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Build/load RAG system
    print("Initializing RAG system...")
    rag = build_rag_system(
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        api_key=api_key,
        force_rebuild=args.rebuild
    )
    print()
    
    # Single question mode
    if args.question:
        answer, sources = rag.answer_question(args.question, k=args.top_k)
        print(f"Question: {args.question}")
        print(f"\nAnswer: {answer}")
        print(f"\nSources: {', '.join(sources)}")
        return
    
    # Interactive mode
    print("=" * 70)
    print("RAG Q&A System - Interactive Mode")
    print("=" * 70)
    print("Ask questions about the Riverside Public Library.")
    print("Type 'quit' or 'exit' to exit.\n")
    
    while True:
        try:
            question = input("❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            answer, sources = rag.answer_question(question, k=args.top_k)
            
            print(f"\n💡 Answer: {answer}")
            print(f"\n📚 Sources: {', '.join(sources)}\n")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == '__main__':
    main()
