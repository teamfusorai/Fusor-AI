Fusor AI – No-Code AI Chatbot Builder
📌 Project Overview

Fusor AI is a web-based No-Code AI Chatbot Builder that enables businesses, educators, and individuals to create, deploy, and manage AI-powered chatbots without writing any code.

The platform allows users to:

Upload documents or provide website URLs

Automatically convert that data into a chatbot knowledge base

Deploy the chatbot as a web widget, QR page, or API

Monitor performance with real-time analytics

Optionally generate AI-powered content from chatbot interactions

The primary goal is to reduce chatbot development time from weeks to under 1 hour for non-technical users.

🎯 Problem Statement

Building AI chatbots today:

Requires coding knowledge

Involves complex integrations

Takes significant development time

Is not accessible to non-technical users

Existing no-code platforms:

Are often too complex

Have rigid workflows

Lack automated knowledge updates

Provide limited customization

Fusor AI solves this by providing a simple, guided, and automated chatbot creation experience.

👥 Target Users
1️⃣ Creator (Primary User)

Business owners

Educators

Teams

Non-technical users

Want instant deployment & analytics

No coding knowledge required

2️⃣ End User

Chats with deployed chatbot

Accesses chatbot via:

Website widget

QR code page

API consumer application

🧠 Core Concept

Fusor AI uses Retrieval-Augmented Generation (RAG) to create context-aware chatbot responses.

Workflow:

User uploads data (PDF, Word, website, etc.)

System:

Extracts text

Chunks content

Generates embeddings

Stores embeddings in vector database

Chatbot retrieves relevant chunks during conversation

LLM generates contextual response

🏗 System Architecture
Frontend

Built with Bubble.io

Provides:

Dashboard

Chatbot creation wizard

Analytics

Customization panel

Backend

Built with FastAPI

Handles:

API endpoints

AI orchestration

Data processing

Authentication

AI Layer

LLMs (OpenAI / Mistral 7B)

LangChain for orchestration

RAG pipeline

Vector Database

Pinecone (free-tier initially)

Stores embeddings for semantic search

Hosting

AWS EC2 (cloud deployment)

🧩 Core Modules
Module 1 – Data Ingestion

Supports:

Website scraping

PDF uploads

Word documents

Spreadsheets

OCR for scanned documents

Processing includes:

Validation

Text chunking

Embedding generation

Storage in vector DB

Module 2 – Knowledge Integration (RAG)

Implements Retrieval-Augmented Generation

Retrieves relevant chunks from vector DB

Generates context-aware answers

Automatically updates knowledge base after new uploads

Sync time: 10–20 minutes

Module 3 – Customization

Users can customize:

Colors

Logos

Fonts

Response tone (formal, casual, technical)

Greeting messages

Fallback responses

Behavior rules

All changes must be previewable.

Module 4 – Web Widget Deployment

Provides embeddable script

Supports:

Floating button

Bottom corner

Inline widget

Embeddable code and domain

Backend serves the widget script at {API_BASE}/static/widget.js and exposes:

GET /embed/config – Returns the public API base URL (domain), widget script URL, placement options, and snippet docs. Use this to “get domain things” (e.g. in Bubble) so the correct API URL is used in the embed snippet.

GET /embed/snippet – Returns the full HTML snippet and api_base_url for a given user_id, bot_id, and optional placement (bottom-corner, floating, inline). Query params: user_id, bot_id, placement, api_url (optional override).

Snippet format: one script tag with data-api-url, data-user-id, data-bot-id, data-placement, and optionally data-target (CSS selector for inline placement).

Domain: Set API_PUBLIC_URL in environment (e.g. https://api.yourdomain.com or your ngrok URL) so the snippet uses a reachable base URL. If unset, the widget uses the script’s origin.

Bubble.io – Deploy / Embed tab: In the Deploy step (or a dedicated “Embed” / “Website widget” tab), (1) call GET /embed/config to get api_public_url and widget_script_url and show the domain to the user; (2) call GET /embed/snippet with query params user_id, bot_id (current chatbot), and optional placement; (3) show the returned snippet in a copy-paste text box. Optionally add a placement dropdown (bottom-corner, floating, inline) and pass it to /embed/snippet. The backend returns the ready-made HTML snippet so Bubble does not need to build it by hand.

Module 5 – API Integration

REST API endpoints

JSON payloads

Sample code:

Python

JavaScript

Secure token-based authentication

Module 6 – QR Code Sharing

Generates QR code linked to chatbot

Enables easy sharing

Requires internet connection

Module 7 – User Interaction Analytics

Tracks:

Total queries

Response time

Success rate

Common questions

Trends over time

Dashboard updates within 5–10 seconds.

Module 8 – Performance Reporting

Daily / Weekly / Monthly reports

Export options:

PDF

CSV

Provides knowledge gap insights

Module 9 – AI Content Generation (Optional)

Automatically generates:

Blog posts

FAQs

Summaries

Troubleshooting guides

Based on:

User queries

Chatbot conversations

Exportable by user.

⚙️ Functional Requirements Summary
ID	Feature	Priority
FR-1	Multi-source data ingestion	High
FR-2	RAG-based knowledge integration	High
FR-3	Chatbot customization	Medium
FR-4	Multi-channel deployment	High
FR-5	Real-time analytics	High
FR-6	AI content generation	Low
📊 Non-Functional Requirements
Performance

95% queries answered in 2–3 seconds

Frontend loads within 4 seconds

Knowledge sync within 10–20 minutes

Reliability

MTBF ≥ 1000 hours

≥ 95% response accuracy

Usability

Chatbot creation within 60 minutes

Core actions reachable within 2 clicks

Security

HTTPS encryption

Secure authentication (OAuth 2.0)

Protected knowledge base access

🔒 System Constraints

Frontend must use Bubble.io

Backend must use FastAPI

Pinecone free-tier (100,000 embeddings limit)

Dependent on OpenAI / Mistral APIs

OCR quality depends on document quality

No offline functionality

No real-time external data fetching

No video/live human chat support

No email/calendar integrations

🚫 Out of Scope

Live human agent handoff

Video communication

CRM/email/calendar integrations

Offline chatbot support

Real-time external web search

📈 Business Objectives

Reduce chatbot build time to < 1 hour

Achieve ≥ 95% response accuracy

Sync new knowledge in < 10 minutes

Enable multi-platform deployment

Provide actionable analytics

🔄 High-Level User Flow

User signs up

Creates new chatbot

Uploads data

System processes & builds knowledge base

User customizes appearance & tone

User previews chatbot

Deploys via:

Widget

API

QR code

Monitors analytics

Improves knowledge iteratively

🧪 Technical Stack
Layer	Technology
Frontend	Bubble.io
Backend	FastAPI
Vector DB	Pinecone
LLM	OpenAI / Mistral 7B
AI Orchestration	LangChain
Hosting	AWS EC2
DevOps	Docker
Version Control	Git
🏁 Vision Statement

Fusor AI aims to democratize AI adoption by making chatbot creation:

Fast

Simple

Accessible

Intelligent

Scalable

The platform removes technical complexity while maintaining advanced AI capabilities.

🧭 Project Category

Web Application

Artificial Intelligence

Generative AI

No-Code Platform

📌 Final Summary

Fusor AI is a:

Web-based, no-code AI chatbot builder that transforms user-provided data into intelligent, context-aware chatbots using Retrieval-Augmented Generation, and enables instant multi-channel deployment with analytics and AI-powered content generation.