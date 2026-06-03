package com.spring.ai.SpringAI;

import java.util.List;

/**
 * Centralised constants for the chat subsystem.
 * Holds provider identifiers, free model lists for OpenRouter, and Google model names.
 */
public final class ChatConstants {

    private ChatConstants() {
        // utility class — no instances
    }

    public static final String PROVIDER_GOOGLE     = "google";
    public static final String PROVIDER_OPENROUTER = "openrouter";
    public static final String PROVIDER_OLLAMA     = "ollama";
    public static final String PROVIDER_AUTO       = "auto";

    /** Default chat provider evaluation order. */
    public static final List<String> DEFAULT_PROVIDER_ORDER = List.of(
            PROVIDER_GOOGLE,
            PROVIDER_OPENROUTER,
            PROVIDER_OLLAMA
    );

    public static final List<String> FREE_OPENROUTER_MODELS = List.of(
            "nvidia/nemotron-nano-9b-v2:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "poolside/laguna-xs.2:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-4-31b-it:free",
            "qwen/qwen3-coder:free"
    );
}
