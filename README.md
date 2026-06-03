# Spring AI Fallback Service

A Spring Boot application integrating multiple Generative AI providers with automatic fallback mechanisms and model rotation to ensure high availability for chat completions and image generation.

## Key Features

- **Robust Chat Service (`ChatService`)**:
  - Orchestrates sequential fallbacks across providers: **Google Gemini** (with optional search grounding) $\rightarrow$ **OpenRouter** (rotating through free models) $\rightarrow$ **Ollama**.
  - Resiliently boots and operates even if certain providers or API credentials are unconfigured or fail.
- **Resilient Image Generation (`ImageGenerationService`)**:
  - Sequences image generation attempts across: **Pollinations** $\rightarrow$ **Hugging Face (Flux.1-schnell)** $\rightarrow$ **Google Gemini (Image Modality)** $\rightarrow$ **OpenRouter**.
  - Automatically handles error responses, rates limits, and switches to fallback models within a provider before trying the next service.
- **REST APIs (`GenAIController`)**:
  - `GET /query-ai`: Queries the chat fallback chain with a text prompt.
  - `GET /query-ai-flash`: Chat query with Google search grounding enabled.
  - `GET /generate-image`: Generates and serves raw image bytes directly.
