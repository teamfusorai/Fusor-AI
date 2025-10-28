#!/usr/bin/env python3
"""
Startup script for Multi-Tenant Chatbot Platform
Runs both Python FastAPI backend and TypeScript frontend
"""

import subprocess
import sys
import os
import time
import threading
import signal
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return the process"""
    print(f"Running: {command}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    process = subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    return process

def monitor_process(process, name):
    """Monitor a process and print its output"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name}] {line.strip()}")
    except Exception as e:
        print(f"Error monitoring {name}: {e}")

def main():
    """Main function to start both services"""
    print("🚀 Starting Multi-Tenant Chatbot Platform")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    frontend_dir = project_root / "Frontend"
    
    # Check if virtual environment exists
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Please run: python -m venv venv")
        sys.exit(1)
    
    # Check if Frontend directory exists
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        print("Please ensure the Frontend directory exists with the TypeScript code")
        sys.exit(1)
    
    # Check if node_modules exists in frontend
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        install_process = run_command("npm install", cwd=frontend_dir)
        install_process.wait()
        if install_process.returncode != 0:
            print("❌ Failed to install frontend dependencies!")
            sys.exit(1)
        print("✅ Frontend dependencies installed!")
    
    processes = []
    
    try:
        # Start Python FastAPI backend
        print("\n🐍 Starting Python FastAPI backend...")
        python_cmd = "venv\\Scripts\\python.exe main.py" if os.name == 'nt' else "venv/bin/python main.py"
        backend_process = run_command(python_cmd, cwd=project_root)
        processes.append(("Backend", backend_process))
        
        # Start monitoring thread for backend
        backend_thread = threading.Thread(
            target=monitor_process, 
            args=(backend_process, "Backend"),
            daemon=True
        )
        backend_thread.start()
        
        # Wait a bit for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Start TypeScript frontend
        print("\n⚛️ Starting TypeScript frontend...")
        frontend_process = run_command("npm run dev", cwd=frontend_dir)
        processes.append(("Frontend", frontend_process))
        
        # Start monitoring thread for frontend
        frontend_thread = threading.Thread(
            target=monitor_process, 
            args=(frontend_process, "Frontend"),
            daemon=True
        )
        frontend_thread.start()
        
        print("\n✅ Both services started successfully!")
        print("=" * 50)
        print("🌐 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("Press Ctrl+C to stop all services")
        
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"❌ {name} process died unexpectedly!")
                    return
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
    finally:
        # Terminate all processes
        for name, process in processes:
            if process.poll() is None:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Force killing {name}...")
                    process.kill()
        
        print("✅ All services stopped!")

if __name__ == "__main__":
    main()
