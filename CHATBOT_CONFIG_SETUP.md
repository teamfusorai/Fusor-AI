# Chatbot Configuration Setup Guide

This guide explains how to configure the chatbot system to fetch configuration from Bubble.io.

## Overview

The system now supports per-chatbot configuration including:
- **System Prompt**: Custom instructions for the LLM
- **Tone**: Response style (professional, casual, friendly, technical, formal, conversational)
- **Industry**: Industry context for better responses
- **Chatbot Name**: Identity of the chatbot
- **Description**: Purpose and description of the chatbot
- **Welcome Message**: Initial greeting (for frontend use)
- **Color & Logo**: UI customization (for frontend use)

## How It Works

1. **Smart System Prompt Generation**: The system automatically builds a comprehensive system prompt using:
   - Custom system prompt (if provided)
   - Chatbot name
   - Description
   - Industry context
   - Tone instructions

2. **Fallback System**: If no custom system prompt is provided, the system creates one using all available chatbot details.

3. **Bubble.io Integration**: Configuration can be fetched from Bubble.io API or passed directly in API requests.

## Environment Variables

Add these to your `.env` file:

```env
# Bubble.io API Configuration
BUBBLE_API_URL=https://your-app.bubbleapps.io/api/1.1
BUBBLE_API_TOKEN=your_bubble_api_token_here
BUBBLE_DATA_TYPE=chatbot  # Your Bubble.io data type name
```

### Getting Bubble.io API Token

1. Go to your Bubble.io app settings
2. Navigate to API → API Token
3. Copy your API token
4. Add it to `.env` as `BUBBLE_API_TOKEN`

## Bubble.io Data Structure

Your Bubble.io data type should have these fields (adjust field names in `chatbot_config.py` if different):

- `Chatbot Name` or `chatbot_name`
- `Description` or `description`
- `Industry` or `industry`
- `Color` or `color`
- `Logo` or `logo`
- `Welcome message` or `welcome_message`
- `Knowledge source` or `knowledge_source`
- `Tone` or `tone`
- `System prompt` or `system_prompt`

Also ensure you have fields to identify the chatbot:
- `user_id` (or similar)
- `bot_id` (or similar)

## API Usage

### Option 1: Fetch Config from Bubble.io (Automatic)

The system will automatically fetch config from Bubble.io if not provided in the request:

```bash
POST /query
{
  "query": "What is this about?",
  "user_id": "user123",
  "bot_id": "bot456",
  "top_k": 3
}
```

### Option 2: Pass Config in Request

You can also pass configuration directly:

```bash
POST /query
{
  "query": "What is this about?",
  "user_id": "user123",
  "bot_id": "bot456",
  "top_k": 3,
  "system_prompt": "You are a helpful assistant...",
  "tone": "professional",
  "industry": "Healthcare",
  "chatbot_name": "HealthBot",
  "description": "A healthcare assistant"
}
```

### Get Chatbot Configuration

```bash
GET /chatbot-config/{user_id}/{bot_id}
```

Returns:
```json
{
  "chatbot_name": "HealthBot",
  "description": "A healthcare assistant",
  "industry": "Healthcare",
  "color": "#3B82F6",
  "logo": "https://...",
  "welcome_message": "Hello! How can I help?",
  "knowledge_source": "Medical documents",
  "tone": "professional",
  "system_prompt": "Custom prompt...",
  "user_id": "user123",
  "bot_id": "bot456"
}
```

## WebSocket Usage

WebSocket endpoint also supports configuration:

```json
{
  "message": "Hello",
  "top_k": 3,
  "system_prompt": "Optional custom prompt",
  "tone": "friendly",
  "industry": "E-commerce"
}
```

## System Prompt Generation

The system automatically builds prompts like this:

```
[Base RAG instructions]

CHATBOT IDENTITY: You are [Chatbot Name].

CHATBOT PURPOSE: [Description]

INDUSTRY CONTEXT: This chatbot serves the [Industry] industry. 
When answering questions, consider industry-specific terminology, 
standards, and best practices relevant to [Industry].

TONE: [Tone-specific instructions]

[Final instructions]
```

## Customization

To customize the Bubble.io API integration, edit `chatbot_config.py`:

1. Adjust field name mappings in the `get_chatbot_config()` function
2. Modify the API URL structure if your Bubble.io setup differs
3. Update constraint field names (`user_id`, `bot_id`) to match your data structure

## Testing Without Bubble.io

If you don't have Bubble.io configured, the system will:
- Use default system prompts
- Accept configuration via API requests
- Work normally without fetching from Bubble.io

Just don't set `BUBBLE_API_URL` and `BUBBLE_API_TOKEN` in your `.env` file.

