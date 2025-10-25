"""
Gradio Interface for Multi-Tenant Chatbot Platform
Provides a user-friendly interface for document upload, ingestion, and chatbot interaction
"""

import gradio as gr
import requests
import json
import os
from typing import List, Tuple, Optional
import config

# API Configuration
API_BASE_URL = "http://localhost:8000"

class ChatbotInterface:
    def __init__(self):
        self.current_user = None
        self.current_bot = None
        self.chat_history = []
    
    def get_available_knowledge_bases(self) -> List[Tuple[str, str]]:
        """Get list of available knowledge bases from Qdrant"""
        try:
            # This would need to be implemented in your FastAPI backend
            # For now, we'll return a placeholder
            response = requests.get(f"{API_BASE_URL}/knowledge-bases")
            if response.status_code == 200:
                kb_list = response.json()
                return [(f"{kb['user_id']} - {kb['bot_id']}", f"{kb['user_id']}|{kb['bot_id']}") for kb in kb_list]
            else:
                return [("No knowledge bases found", "")]
        except Exception as e:
            print(f"Error fetching knowledge bases: {e}")
            return [("Error loading knowledge bases", "")]
    
    def upload_and_ingest_documents(self, username: str, bot_name: str, files: List[str]) -> str:
        """Upload documents and trigger ingestion pipeline"""
        if not username or not bot_name:
            return "❌ Please enter both username and bot name"
        
        if not files:
            return "❌ Please upload at least one document"
        
        try:
            results = []
            for file_path in files:
                with open(file_path, 'rb') as f:
                    files_data = {'file': f}
                    data = {
                        'user_id': username,
                        'bot_id': bot_name
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/ingest",
                        files=files_data,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results.append(f"✅ {os.path.basename(file_path)}: {result['chunks_stored']} chunks stored")
                    else:
                        results.append(f"❌ {os.path.basename(file_path)}: {response.text}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"❌ Error during ingestion: {str(e)}"
    
    def query_chatbot(self, message: str, username: str, bot_name: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """Query the chatbot and return response"""
        if not message.strip():
            return "", history
        
        if not username or not bot_name:
            return "❌ Please enter both username and bot name", history
        
        try:
            # Prepare request
            payload = {
                "query": message,
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
                answer = result.get("answer", "No response received")
                citations = result.get("citations", [])
                chunks_used = result.get("chunks_used", 0)
                
                # Format response with citations
                if citations:
                    answer += f"\n\n📚 Sources: {'; '.join(citations)}"
                
                # Add metadata
                answer += f"\n\n📊 Used {chunks_used} relevant chunks"
                
                # Update history with message format
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": answer})
                
                return "", history
            else:
                error_msg = f"❌ Error: {response.text}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                return "", history
                
        except Exception as e:
            error_msg = f"❌ Connection error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    def select_knowledge_base(self, kb_selection: str) -> Tuple[str, str]:
        """Handle knowledge base selection"""
        if kb_selection and "|" in kb_selection:
            user_id, bot_id = kb_selection.split("|", 1)
            return user_id, bot_id
        return "", ""

# Initialize interface
chatbot_interface = ChatbotInterface()

def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="Multi-Tenant Chatbot Platform", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Multi-Tenant Chatbot Platform")
        gr.Markdown("Upload documents to create knowledge bases and chat with your personalized AI assistants.")
        
        with gr.Tabs():
            # Tab 1: Document Upload & Ingestion
            with gr.Tab("📁 Upload Documents"):
                gr.Markdown("## Create a New Knowledge Base")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        username_input = gr.Textbox(
                            label="Username",
                            placeholder="Enter your username"
                        )
                        bot_name_input = gr.Textbox(
                            label="Bot Name",
                            placeholder="Enter bot name"
                        )
                        
                        file_upload = gr.File(
                            label="Upload Documents",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"]
                        )
                        
                        upload_btn = gr.Button("🚀 Upload & Process Documents", variant="primary")
                    
                    with gr.Column(scale=1):
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            lines=10,
                            interactive=False,
                            placeholder="Upload status will appear here..."
                        )
                
                upload_btn.click(
                    fn=chatbot_interface.upload_and_ingest_documents,
                    inputs=[username_input, bot_name_input, file_upload],
                    outputs=upload_status
                )
            
            # Tab 2: Chat Interface
            with gr.Tab("💬 Chat with Bot"):
                gr.Markdown("## Chat with Your AI Assistant")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Knowledge Base Selection
                        with gr.Row():
                            kb_dropdown = gr.Dropdown(
                                label="Select Knowledge Base",
                                choices=[],
                                interactive=True
                            )
                            refresh_kb_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        # Manual User/Bot Input
                        with gr.Row():
                            chat_username = gr.Textbox(
                                label="Username",
                                placeholder="Enter username",
                                scale=1
                            )
                            chat_botname = gr.Textbox(
                                label="Bot Name",
                                placeholder="Enter bot name",
                                scale=1
                            )
                        
                        # Chat Interface
                        chatbot = gr.Chatbot(
                            label="Chat History",
                            height=400,
                            show_label=True,
                            type="messages"
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Your Message",
                                placeholder="Type your question here...",
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Column(scale=1):
                        # Current Knowledge Base Info
                        gr.Markdown("### 📚 Current Knowledge Base")
                        current_kb_info = gr.Textbox(
                            label="Active Knowledge Base",
                            value="No knowledge base selected",
                            interactive=False,
                            lines=3
                        )
                        
                        # Knowledge Base Statistics
                        gr.Markdown("### 📊 Knowledge Base Stats")
                        kb_stats = gr.Textbox(
                            label="Statistics",
                            value="No data available",
                            interactive=False,
                            lines=5
                        )
                        
                        # Available Knowledge Bases List
                        gr.Markdown("### 🗂️ Available Knowledge Bases")
                        available_kbs = gr.Textbox(
                            label="All Knowledge Bases",
                            value="Loading...",
                            interactive=False,
                            lines=8
                        )
                
                # Event handlers
                def refresh_knowledge_bases():
                    """Refresh the knowledge base dropdown and list"""
                    kb_choices = chatbot_interface.get_available_knowledge_bases()
                    kb_list_text = "\n".join([f"• {choice[0]}" for choice in kb_choices if choice[0] != "No knowledge bases found"])
                    
                    return gr.Dropdown(choices=kb_choices), kb_list_text
                
                def update_kb_selection(kb_selection):
                    """Update the current knowledge base selection"""
                    if kb_selection and "|" in kb_selection:
                        user_id, bot_id = kb_selection.split("|", 1)
                        kb_info = f"User: {user_id}\nBot: {bot_id}\nStatus: Active"
                        return user_id, bot_id, kb_info
                    return "", "", "No knowledge base selected"
                
                def update_chat_kb_info(username, botname):
                    """Update knowledge base info when manual input is used"""
                    if username and botname:
                        return f"User: {username}\nBot: {botname}\nStatus: Active"
                    return "No knowledge base selected"
                
                # Connect event handlers
                refresh_kb_btn.click(
                    fn=refresh_knowledge_bases,
                    outputs=[kb_dropdown, available_kbs]
                )
                
                kb_dropdown.change(
                    fn=update_kb_selection,
                    inputs=kb_dropdown,
                    outputs=[chat_username, chat_botname, current_kb_info]
                )
                
                chat_username.change(
                    fn=update_chat_kb_info,
                    inputs=[chat_username, chat_botname],
                    outputs=current_kb_info
                )
                
                chat_botname.change(
                    fn=update_chat_kb_info,
                    inputs=[chat_username, chat_botname],
                    outputs=current_kb_info
                )
                
                send_btn.click(
                    fn=chatbot_interface.query_chatbot,
                    inputs=[msg_input, chat_username, chat_botname, chatbot],
                    outputs=[msg_input, chatbot]
                )
                
                msg_input.submit(
                    fn=chatbot_interface.query_chatbot,
                    inputs=[msg_input, chat_username, chat_botname, chatbot],
                    outputs=[msg_input, chatbot]
                )
                
                # Initialize on load
                demo.load(
                    fn=refresh_knowledge_bases,
                    outputs=[kb_dropdown, available_kbs]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("Built with ❤️ using Gradio and FastAPI")
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
