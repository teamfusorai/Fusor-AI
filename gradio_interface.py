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
    
    def upload_and_ingest_documents(self, username: str, bot_name: str, files: List[str], url: str, url_type: str) -> str:
        """Upload documents or URLs and trigger ingestion pipeline"""
        if not username or not bot_name:
            return "❌ Please enter both username and bot name"
        
        if not files and not url:
            return "❌ Please upload documents OR enter a website URL"
        
        if files and url:
            return "❌ Please choose either file upload OR URL processing, not both"
        
        try:
            results = []
            
            # Process files
            if files:
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
            
            # Process URL
            elif url:
                # Validate URL format
                if not url.startswith(('http://', 'https://')):
                    return "❌ Please enter a valid URL starting with http:// or https://"
                
                # Prepare URL for processing
                if url_type == "Entire Website (Sitemap)":
                    # Add sitemap.xml if not present
                    if not url.endswith('sitemap.xml'):
                        url = url.rstrip('/') + '/sitemap.xml'
                
                data = {
                    'url': url,
                    'user_id': username,
                    'bot_id': bot_name
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/ingest",
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if url_type == "Entire Website (Sitemap)":
                        results.append(f"✅ Website processed: {url}")
                        results.append(f"📊 {result['chunks_stored']} chunks stored from entire website")
                        results.append(f"ℹ️ Note: If sitemap was blocked, only the main page was processed")
                    else:
                        results.append(f"✅ Page processed: {url}")
                        results.append(f"📊 {result['chunks_stored']} chunks stored")
                else:
                    results.append(f"❌ URL processing failed: {response.text}")
                    results.append(f"💡 Try using 'Single Page' mode instead of 'Entire Website'")
            
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
    
    # Custom CSS for dark theme matching the design
    custom_css = """
    :root {
        --body-background-fill: #0a0a0b;
        --background-fill-primary: rgba(24, 24, 27, 0.5);
        --background-fill-secondary: rgba(24, 24, 27, 0.3);
        --border-color-primary: #27272a;
        --text-color-primary: #ffffff;
        --text-color-secondary: #a1a1aa;
        --color-accent: #4f46e5;
        --color-accent-soft: rgba(79, 70, 229, 0.5);
        --radius-lg: 8px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
    }
    
    .gradio-container {
        background: var(--body-background-fill) !important;
        color: var(--text-color-primary) !important;
    }
    
    .tab-nav {
        border-bottom: 1px solid var(--border-color-primary) !important;
    }
    
    .tab-nav button {
        background: transparent !important;
        border: none !important;
        color: #71717a !important;
        padding: 12px 16px !important;
        transition: color 0.2s !important;
    }
    
    .tab-nav button.selected {
        color: var(--text-color-primary) !important;
        border-bottom: 2px solid var(--color-accent) !important;
    }
    
    .tab-nav button:hover {
        color: #d4d4d8 !important;
    }
    
    .gr-textbox, .gr-dropdown, .gr-file {
        background: var(--background-fill-primary) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: var(--radius-lg) !important;
        color: var(--text-color-primary) !important;
    }
    
    .gr-textbox:focus, .gr-dropdown:focus {
        outline: none !important;
        border-color: transparent !important;
        box-shadow: 0 0 0 2px var(--color-accent-soft) !important;
    }
    
    .gr-button {
        border-radius: var(--radius-lg) !important;
        transition: all 0.2s !important;
    }
    
    .gr-button.primary {
        background: var(--color-accent) !important;
        color: white !important;
        border: none !important;
    }
    
    .gr-button.primary:hover {
        background: #4338ca !important;
    }
    
    .gr-button.secondary {
        background: var(--background-fill-primary) !important;
        border: 1px solid var(--border-color-primary) !important;
        color: #d4d4d8 !important;
    }
    
    .gr-button.secondary:hover {
        border-color: #3f3f46 !important;
    }
    
    .gr-file {
        border: 2px dashed var(--border-color-primary) !important;
        padding: 48px !important;
        text-align: center !important;
    }
    
    .gr-file:hover {
        border-color: var(--color-accent) !important;
        background: rgba(79, 70, 229, 0.05) !important;
    }
    
    .gr-chatbot {
        background: var(--background-fill-secondary) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: var(--radius-lg) !important;
    }
    
    .info-box {
        background: rgba(245, 158, 11, 0.05) !important;
        border: 1px solid rgba(245, 158, 11, 0.2) !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--spacing-md) !important;
        margin: var(--spacing-md) 0 !important;
    }
    
    .info-box h4 {
        color: #f59e0b !important;
        margin: 0 0 8px 0 !important;
    }
    
    .info-box ul {
        color: var(--text-color-secondary) !important;
        margin: 0 !important;
        font-size: 0.9rem !important;
    }
    
    .card {
        background: var(--background-fill-secondary) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: var(--radius-lg) !important;
        padding: var(--spacing-lg) !important;
        margin-bottom: var(--spacing-md) !important;
    }
    
    .card h3 {
        color: var(--text-color-primary) !important;
        margin: 0 0 var(--spacing-md) 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: var(--spacing-sm) !important;
    }
    
    .card .status-active {
        color: #10b981 !important;
    }
    
    .card .status-inactive {
        color: var(--text-color-secondary) !important;
    }
    
    .header {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        padding: var(--spacing-md) 0 !important;
        margin-bottom: var(--spacing-lg) !important;
    }
    
    .header-left {
        display: flex !important;
        align-items: center !important;
        gap: var(--spacing-sm) !important;
    }
    
    .header-right {
        display: flex !important;
        align-items: center !important;
        gap: var(--spacing-md) !important;
    }
    
    .logo {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        color: var(--text-color-primary) !important;
    }
    
    .user-dropdown {
        color: var(--text-color-primary) !important;
        background: transparent !important;
        border: none !important;
        cursor: pointer !important;
    }
    
    .share-btn {
        background: var(--color-accent) !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: var(--radius-lg) !important;
        cursor: pointer !important;
    }
    
    .upload-zone {
        border: 2px dashed var(--border-color-primary) !important;
        border-radius: var(--radius-lg) !important;
        padding: 48px !important;
        text-align: center !important;
        background: transparent !important;
        transition: all 0.2s !important;
    }
    
    .upload-zone:hover {
        border-color: var(--color-accent) !important;
        background: rgba(79, 70, 229, 0.05) !important;
    }
    
    .upload-icon {
        font-size: 2.5rem !important;
        color: #71717a !important;
        margin-bottom: var(--spacing-md) !important;
    }
    
    .radio-group {
        display: flex !important;
        gap: var(--spacing-sm) !important;
        margin: var(--spacing-md) 0 !important;
    }
    
    .radio-button {
        background: var(--background-fill-primary) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: var(--radius-lg) !important;
        padding: 8px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        color: var(--text-color-secondary) !important;
    }
    
    .radio-button.selected {
        background: var(--color-accent) !important;
        color: white !important;
        border-color: var(--color-accent) !important;
    }
    
    .radio-button:hover {
        border-color: #3f3f46 !important;
    }
    """
    
    with gr.Blocks(
        title="Multi-Tenant Chatbot Platform", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        with gr.Row(elem_classes="header"):
            with gr.Column(scale=1, elem_classes="header-left"):
                gr.HTML('<div class="logo">AI</div>')
            with gr.Column(scale=1, elem_classes="header-right"):
                gr.HTML('<div class="user-dropdown">User Introduction ▼</div>')
                gr.HTML('<div class="share-btn">Share</div>')
        
        # Main Title
        gr.Markdown("# Multi-Tenant Chatbot Platform")
        gr.Markdown("Upload documents to create knowledge bases and chat with your personalized AI assistants.")
        
        with gr.Tabs(elem_classes="tab-nav"):
            # Tab 1: Document Upload & Ingestion
            with gr.Tab("Upload Documents"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Create a New Knowledge Base")
                        
                        username_input = gr.Textbox(
                            label="Username",
                            placeholder="Enter your username",
                            elem_classes="gr-textbox"
                        )
                        
                        bot_name_input = gr.Textbox(
                            label="Bot Name",
                            placeholder="Enter bot name",
                            elem_classes="gr-textbox"
                        )
                        
                        gr.Markdown("**Upload Documents**")
                        file_upload = gr.File(
                            label="",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                            elem_classes="gr-file upload-zone"
                        )
                        
                        gr.HTML("""
                        <div style="text-align: center; margin: 16px 0; color: #ffffff;">
                            <strong>OR</strong>
                        </div>
                        """)
                        
                        url_input = gr.Textbox(
                            label="Website URL",
                            placeholder="Enter website URL (e.g., https://example.com)",
                            info="Supports single pages or sitemap.xml for entire websites",
                            elem_classes="gr-textbox"
                        )
                        
                        gr.Markdown("**URL Type**")
                        gr.Markdown("*Single page, Process one webpage | Sitemap: Process entire website*")
                        
                        url_type = gr.Radio(
                            choices=["Single Page", "Entire Website (Sitemap)"],
                            value="Single Page",
                            label="",
                            elem_classes="radio-group"
                        )
                        
                        gr.HTML("""
                        <div class="info-box">
                            <h4>URL Processing Tips:</h4>
                            <ul>
                                <li>For single pages, enter the complete URL</li>
                                <li>For sitemaps, ensure sitemap.xml is accessible</li>
                                <li>Large websites may take longer to process</li>
                            </ul>
                        </div>
                        """)
                        
                        upload_btn = gr.Button("Create Knowledge Base", variant="primary", elem_classes="gr-button primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("## Upload Status")
                        upload_status = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False,
                            placeholder="Upload status will appear here...",
                            elem_classes="gr-textbox"
                        )
                
                upload_btn.click(
                    fn=chatbot_interface.upload_and_ingest_documents,
                    inputs=[username_input, bot_name_input, file_upload, url_input, url_type],
                    outputs=upload_status
                )
            
            # Tab 2: Chat Interface
            with gr.Tab("Chat with Bot"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## Chat with Your AI Assistant")
                        
                        # Knowledge Base Selection
                            kb_dropdown = gr.Dropdown(
                            label="Select Knowledge Base...",
                                choices=[],
                            interactive=True,
                            elem_classes="gr-dropdown"
                            )
                        
                        # Manual User/Bot Input
                        with gr.Row():
                            chat_username = gr.Textbox(
                                label="Username",
                                placeholder="Enter username",
                                scale=1,
                                elem_classes="gr-textbox"
                            )
                            chat_botname = gr.Textbox(
                                label="Bot Name",
                                placeholder="Enter bot name",
                                scale=1,
                                elem_classes="gr-textbox"
                            )
                            refresh_kb_btn = gr.Button("🔄", elem_classes="gr-button secondary")
                        
                        # Chat Interface
                        chatbot = gr.Chatbot(
                            label="Chat History",
                            height=400,
                            show_label=True,
                            type="messages",
                            elem_classes="gr-chatbot",
                            placeholder="Start a conversation..."
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Your Message",
                                placeholder="Type your question here...",
                                scale=4,
                                elem_classes="gr-textbox"
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes="gr-button primary")
                    
                    with gr.Column(scale=1):
                        # Current Knowledge Base Info
                        with gr.Group(elem_classes="card"):
                            gr.HTML('<h3>📊 Current Knowledge Base</h3>')
                        current_kb_info = gr.Textbox(
                                label="",
                            value="No knowledge base selected",
                            interactive=False,
                                lines=3,
                                elem_classes="gr-textbox"
                        )
                        
                        # Knowledge Base Statistics
                        with gr.Group(elem_classes="card"):
                            gr.HTML('<h3>📈 Knowledge Base Stats</h3>')
                        kb_stats = gr.Textbox(
                                label="",
                            value="No data available",
                            interactive=False,
                                lines=5,
                                elem_classes="gr-textbox"
                        )
                        
                        # Available Knowledge Bases List
                        with gr.Group(elem_classes="card"):
                            gr.HTML('<h3>🗂️ Available Knowledge Bases</h3>')
                        available_kbs = gr.Textbox(
                                label="",
                            value="Loading...",
                            interactive=False,
                                lines=8,
                                elem_classes="gr-textbox"
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
        gr.Markdown("Built with Gradio and FastAPI")
    
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
