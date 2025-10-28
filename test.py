import os
import requests
from dotenv import load_dotenv
load_dotenv()

# API Configuration
API_BASE_URL = "http://localhost:8000"

def query_knowledge_base(query: str, username: str, bot_name: str):
    """Query the knowledge base using the RAG pipeline"""
    try:
        # Prepare request
        payload = {
            "query": query,
            "user_id": username,
            "bot_id": bot_name,
            "top_k": 3
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {"error": f"API Error: {response.text}"}
            
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def test_knowledge_base():
    """Test the knowledge base with sample queries"""
    
    # Configuration
    username = "anas183"  # Change this to your username
    bot_name = "bot1"     # Change this to your bot name
    
    # Test queries
    test_queries = [
        "What is the main difference between HTTP and HTTPS?",
        "What are the key features of this system?",
        "Can you summarize the main points?",
        "What technologies are used in this project?"
    ]
    
    print(f"🤖 Testing Knowledge Base: {username}/{bot_name}")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        # Query the knowledge base
        result = query_knowledge_base(query, username, bot_name)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            # Display results
            answer = result.get("answer", "No response received")
            citations = result.get("citations", [])
            chunks_used = result.get("chunks_used", 0)
            confidence_scores = result.get("confidence_scores", [])
            
            print(f"✅ Answer: {answer}")
            
            if citations:
                print(f"📚 Sources: {'; '.join(citations)}")
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                print(f"📊 Confidence: {avg_confidence:.2f} (used {chunks_used} chunks)")
            else:
                print(f"📊 Used {chunks_used} chunks")

if __name__ == "__main__":
    test_knowledge_base()
