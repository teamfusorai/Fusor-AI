// API service for communicating with Python FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: string[];
  chunks_used?: number;
}

export interface KnowledgeBase {
  user_id: string;
  bot_id: string;
  chunks_count: number;
  created_at?: string;
}

export interface UploadResponse {
  status: string;
  chunks_stored: number;
  qdrant_collection: string;
  namespace: string;
  user_id: string;
  bot_id: string;
  source_name: string;
}

export interface QueryResponse {
  answer: string;
  citations: string[];
  chunks_used: number;
  confidence_scores?: number[];
}

export interface ApiError {
  error: string;
  detail?: string;
}

export class ChatbotAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Upload documents to create a knowledge base
   */
  async uploadDocuments(
    files: File[],
    username: string,
    botName: string
  ): Promise<UploadResponse> {
    const formData = new FormData();
    
    // Add files to form data
    files.forEach((file) => {
      formData.append('file', file);
    });
    
    // Add user and bot information
    formData.append('user_id', username);
    formData.append('bot_id', botName);

    const response = await fetch(`${this.baseUrl}/ingest`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new Error(error.error || 'Upload failed');
    }

    return response.json();
  }

  /**
   * Upload from URL to create a knowledge base
   */
  async uploadFromUrl(
    url: string,
    urlType: 'single' | 'sitemap',
    username: string,
    botName: string
  ): Promise<UploadResponse> {
    const formData = new FormData();
    
    // Add URL and type information
    formData.append('url', url);
    formData.append('user_id', username);
    formData.append('bot_id', botName);

    const response = await fetch(`${this.baseUrl}/ingest`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new Error(error.error || 'URL upload failed');
    }

    return response.json();
  }

  /**
   * Query the chatbot
   */
  async queryChatbot(
    message: string,
    username: string,
    botName: string,
    topK: number = 3
  ): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: message,
        user_id: username,
        bot_id: botName,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new Error(error.error || 'Query failed');
    }

    return response.json();
  }

  /**
   * Get list of available knowledge bases
   */
  async getKnowledgeBases(): Promise<KnowledgeBase[]> {
    const response = await fetch(`${this.baseUrl}/knowledge-bases`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new Error(error.error || 'Failed to fetch knowledge bases');
    }

    return response.json();
  }

  /**
   * Get knowledge base statistics
   */
  async getKnowledgeBaseStats(
    username: string,
    botName: string
  ): Promise<{ chunks_count: number; last_updated?: string }> {
    const response = await fetch(
      `${this.baseUrl}/knowledge-bases/${username}/${botName}/stats`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new Error(error.error || 'Failed to fetch stats');
    }

    return response.json();
  }

  /**
   * Delete a knowledge base
   */
  async deleteKnowledgeBase(
    username: string,
    botName: string
  ): Promise<{ status: string }> {
    const response = await fetch(
      `${this.baseUrl}/knowledge-bases/${username}/${botName}`,
      {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new Error(error.error || 'Failed to delete knowledge base');
    }

    return response.json();
  }
}

// WebSocket service for real-time chat
export class ChatWebSocket {
  private ws: WebSocket | null = null;
  private onMessage: ((data: QueryResponse) => void) | null = null;
  private onError: ((error: Event) => void) | null = null;
  private onClose: (() => void) | null = null;

  constructor(
    onMessage: (data: QueryResponse) => void,
    onError?: (error: Event) => void,
    onClose?: () => void
  ) {
    this.onMessage = onMessage;
    this.onError = onError;
    this.onClose = onClose;
  }

  connect(userId: string, botId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`ws://localhost:8000/ws/chat/${userId}/${botId}`);
        
        this.ws.onopen = () => {
          console.log('WebSocket connected');
          resolve();
        };
        
        this.ws.onmessage = (event) => {
          try {
            const data: QueryResponse = JSON.parse(event.data);
            this.onMessage?.(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.onError?.(error);
          reject(error);
        };
        
        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.onClose?.();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  sendMessage(message: string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ message }));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Export singleton instance
export const api = new ChatbotAPI();
