package com.spring.ai.SpringAI;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.google.genai.GoogleGenAiChatModel;
import org.springframework.ai.google.genai.GoogleGenAiChatOptions;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.spring.ai.SpringAI.ChatConstants.*;

/**
 * Orchestrates chat completions across multiple providers with automatic
 * fallback.  Provider order: Google Gemini → OpenRouter (free models) → Ollama.
 *
 * <p>Each provider is injected as an {@link ObjectProvider} so the application
 * starts even when a provider's auto-configuration is missing or incomplete.</p>
 */
@Service
public class ChatService {

    private static final Logger logger = LoggerFactory.getLogger(ChatService.class);

    private final GoogleGenAiChatModel googleModel;
    private final OpenAiChatModel openRouterModel;   // OpenAI-compatible, pointed at OpenRouter
    private final OllamaChatModel ollamaModel;

    public ChatService(
            ObjectProvider<GoogleGenAiChatModel> googleProvider,
            ObjectProvider<OpenAiChatModel> openAiProvider,
            ObjectProvider<OllamaChatModel> ollamaProvider) {
        this.googleModel     = googleProvider.getIfAvailable();
        this.openRouterModel = openAiProvider.getIfAvailable();
        this.ollamaModel     = ollamaProvider.getIfAvailable();

        logger.info("Chat providers available — Google: {}, OpenRouter: {}, Ollama: {}",
                googleModel != null, openRouterModel != null, ollamaModel != null);
    }

    /**
     * Basic chat completion — tries every configured provider in order.
     */
    public String getResponse(String prompt) {
        return executeWithFallback(prompt, null);
    }

    /**
     * Chat completion with Google-specific options (search grounding, Flash Lite
     * model, higher temperature).  If Google is unavailable the grounding options
     * are silently dropped and the next provider is tried with a plain prompt.
     */
    public String getResponseOptions(String prompt) {
        GoogleGenAiChatOptions googleOptions = GoogleGenAiChatOptions.builder()
                .model(GoogleGenAiChatModel.ChatModel.GEMINI_3_1_FLASH_LITE)
                .googleSearchRetrieval(true)
                .temperature(0.9)
                .build();

        return executeWithFallback(prompt, googleOptions);
    }

    /**
     * Walks through the provider sequence, returning the first successful
     * response.  Google-specific {@code googleOptions} are only applied when
     * the Google provider is attempted; all other providers receive a plain
     * prompt (optionally with model overrides).
     */
    private String executeWithFallback(String prompt, GoogleGenAiChatOptions googleOptions) {
        List<String> sequence = DEFAULT_PROVIDER_ORDER;
        List<Throwable> errors = new ArrayList<>();

        for (String provider : sequence) {
            if (!isProviderAvailable(provider)) {
                logger.info("Skipping chat provider '{}' — not available.", provider);
                continue;
            }
            try {
                logger.info("Attempting chat via '{}'…", provider);
                String result = dispatchToProvider(provider, prompt, googleOptions);
                logger.info("Chat succeeded via '{}'.", provider);
                return result;
            } catch (Exception ex) {
                logger.warn("Chat provider '{}' failed: {}", provider, ex.getMessage());
                errors.add(ex);
            }
        }

        ChatException allFailed = new ChatException(
                "All chat providers failed to generate a response.", null);
        errors.forEach(allFailed::addSuppressed);
        throw allFailed;
    }

    private String dispatchToProvider(String provider, String prompt,
                                      GoogleGenAiChatOptions googleOptions) {
        return switch (provider) {
            case PROVIDER_GOOGLE     -> callGoogle(prompt, googleOptions);
            case PROVIDER_OPENROUTER -> callOpenRouterWithFallback(prompt);
            case PROVIDER_OLLAMA     -> callOllama(prompt);
            default -> throw new IllegalStateException("Unknown chat provider: " + provider);
        };
    }

    private String callGoogle(String prompt, GoogleGenAiChatOptions options) {
        if (options != null) {
            ChatResponse response = googleModel.call(new Prompt(prompt, options));
            return extractText(response);
        }
        return googleModel.call(prompt);
    }

    /**
     * Tries each free OpenRouter model in order until one succeeds.
     */
    private String callOpenRouterWithFallback(String prompt) {
        List<Throwable> errors = new ArrayList<>();

        for (String model : FREE_OPENROUTER_MODELS) {
            try {
                logger.info("OpenRouter → trying free model '{}'…", model);
                OpenAiChatOptions options = OpenAiChatOptions.builder()
                        .model(model)
                        .temperature(0.2)
                        .build();

                ChatResponse response = openRouterModel.call(new Prompt(prompt, options));
                String text = extractText(response);
                logger.info("OpenRouter → model '{}' succeeded.", model);
                return text;
            } catch (Exception ex) {
                logger.warn("OpenRouter model '{}' failed: {}", model, ex.getMessage());
                errors.add(ex);
            }
        }

        ChatException allFailed = new ChatException(
                "All free OpenRouter models failed.", null);
        errors.forEach(allFailed::addSuppressed);
        throw allFailed;
    }

    private String callOllama(String prompt) {
        return ollamaModel.call(prompt);
    }

    private boolean isProviderAvailable(String provider) {
        return switch (provider) {
            case PROVIDER_GOOGLE     -> googleModel != null;
            case PROVIDER_OPENROUTER -> openRouterModel != null;
            case PROVIDER_OLLAMA     -> ollamaModel != null;
            default -> false;
        };
    }

    private String extractText(ChatResponse response) {
        return Objects.requireNonNull(
                response.getResult(), "Chat response had no result")
                .getOutput().getText();
    }
}
