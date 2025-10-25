"""
Startup script for Multi-Tenant Chatbot Platform
Runs both FastAPI backend and Gradio interface
"""

import subprocess
import time
import sys
import os
from threading import Thread

def run_fastapi():
    """Run the FastAPI backend"""
    print("🚀 Starting FastAPI backend on http://localhost:8000")
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    except KeyboardInterrupt:
        print("FastAPI backend stopped")

def run_gradio():
    """Run the Gradio interface"""
    print("⏳ Waiting for FastAPI backend to start...")
    time.sleep(8)  # Wait longer for FastAPI to start
    
    print("🎨 Starting Gradio interface on http://localhost:7860")
    try:
        subprocess.run([sys.executable, "gradio_interface.py"])
    except KeyboardInterrupt:
        print("Gradio interface stopped")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🤖 Multi-Tenant Chatbot Platform")
    print("=" * 60)
    print("Starting both FastAPI backend and Gradio interface...")
    print("")
    print("📡 FastAPI Backend: http://localhost:8000")
    print("🎨 Gradio Interface: http://localhost:7860")
    print("")
    print("Press Ctrl+C to stop both services")
    print("=" * 60)
    
    # Start FastAPI in a separate thread
    fastapi_thread = Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Start Gradio in the main thread
    try:
        run_gradio()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down platform...")
        print("✅ Platform stopped successfully")

if __name__ == "__main__":
    main()
