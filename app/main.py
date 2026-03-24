"""
main.py

Entry point for running and testing the chatbot locally.
 
Usage:
    python main.py
 
Loads the vector DB and starts an interactive loop where you can type
questions and see the answers. Type 'quit' or 'exit' to stop.

"""

from vectorstore import load_db
from retrieval import query_answer_pipe

def main():
    print("=" * 50)
    print( "PVL by-laws Chatbot")
    print("=" * 50)
    print("Loading database...")

    db = load_db()

    print("\nReady. Type your question below.")
    print("Type 'quit' to exit.\n")

    while True:

        query = input("Your question:").strip()

        if not query:
            continue

        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            break
        
        print("\n Searching...\n")
        answer = query_answer_pipe(db,query)

        print(f"Answer: \n{answer}")
        print("=" * 50 + "\n")













if __name__ == "__main__":
    main()