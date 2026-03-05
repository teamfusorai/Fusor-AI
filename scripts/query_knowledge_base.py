"""
Manual script to query the knowledge base via the API.
Run from project root: python scripts/query_knowledge_base.py
"""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def query_knowledge_base(query: str, username: str, bot_name: str):
    """Query the knowledge base using the RAG pipeline"""
    try:
        payload = {
            "query": query,
            "user_id": username,
            "bot_id": bot_name,
            "top_k": 3
        }
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.text}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


def main():
    username = os.getenv("TEST_USER_ID", "anas183")
    bot_name = os.getenv("TEST_BOT_ID", "bot1")
    test_queries = [
        "What is the main difference between HTTP and HTTPS?",
        "What are the key features of this system?",
        "Can you summarize the main points?",
        "What technologies are used in this project?",
    ]
    print(f"Testing Knowledge Base: {username}/{bot_name}")
    print("=" * 60)
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        result = query_knowledge_base(query, username, bot_name)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Answer: {result.get('answer', 'No response')}")
            if result.get("citations"):
                print(f"Sources: {'; '.join(result['citations'])}")
            print(f"Chunks used: {result.get('chunks_used', 0)}")


if __name__ == "__main__":
    main()
