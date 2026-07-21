"""
STKG CLI - Spatio-Temporal Knowledge Graph Query System
LLM-based Automatic Construction Framework

Usage:
    python cli.py
"""

import sys
sys.path.append('.')
from modules.query import STKGQueryEngine

def print_banner():
    print("=" * 60)
    print("  STKG CLI - Spatio-Temporal Knowledge Graph Query System")
    print("  LLM-based Automatic Construction Framework")
    print("  Korea Maritime & Ocean University - Wangjae Rhee")
    print("=" * 60)
    print()

def print_help():
    print("Usage:")
    print("  Enter question  → Natural language answer based on STKG")
    print("  /stats          → STKG statistics")
    print("  /help           → Show help")
    print("  /exit           → Exit")
    print()
    print("Example queries:")
    print("  What did Saudi Arabia do on 2022-10-13?")
    print("  What actions did Israel take?")
    print("  What events occurred in the United States?")
    print("  Which country was mentioned the most?")
    print()

def main():
    print_banner()
    print("Connecting to Neo4j...")

    try:
        engine = STKGQueryEngine()
        print("Connected!\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print_help()

    while True:
        try:
            user_input = input("Query > ").strip()

            if not user_input:
                continue

            if user_input.lower() == '/exit':
                print("Exiting...")
                engine.close()
                break

            elif user_input.lower() == '/help':
                print_help()

            elif user_input.lower() == '/stats':
                stats = engine.builder.get_stats()
                print(f"\nSTKG Statistics")
                print(f"  Number of nodes: {stats['nodes']}")
                print(f"  Number of edges: {stats['edges']}\n")

            else:
                print("\nSearching...\n")
                answer, results, parsed = engine.query(user_input)

                print(f"[Parsed Result] {parsed}")
                print(f"[STKG Search Results: {len(results)} items]")
                for r in results[:5]:
                    print(f"  {r['entity1']} → {r['relation']} → "
                          f"{r['entity2']} ({r['time']}, {r['location']})")

                print(f"\n[Answer]")
                print(f"{answer}\n")
                print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            engine.close()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()