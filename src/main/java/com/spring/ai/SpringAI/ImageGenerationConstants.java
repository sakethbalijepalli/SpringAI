package com.spring.ai.SpringAI;

import java.time.Duration;
import java.util.List;

/**
 * Centralised constants for the image generation subsystem.
 * Keeps magic strings, URLs, timeouts, and fallback model lists out of the service logic.
 */
public final class ImageGenerationConstants {

    private ImageGenerationConstants() {
        // utility class — no instances
    }

    public static final String PROVIDER_GOOGLE       = "google";
    public static final String PROVIDER_OPENROUTER   = "openrouter";
    public static final String PROVIDER_HUGGINGFACE   = "huggingface";
    public static final String PROVIDER_POLLINATIONS  = "pollinations";
    public static final String PROVIDER_AUTO          = "auto";

    /** Default provider evaluation order when in auto mode. */
    public static final List<String> DEFAULT_PROVIDER_ORDER = List.of(
            PROVIDER_GOOGLE,
            PROVIDER_OPENROUTER,
            PROVIDER_HUGGINGFACE,
            PROVIDER_POLLINATIONS
    );

    public static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(15);
    public static final Duration READ_TIMEOUT    = Duration.ofSeconds(120);

    public static final String GOOGLE_GENERATE_URI =
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}";
    public static final String OPENROUTER_COMPLETIONS_PATH = "/chat/completions";
    public static final String HUGGINGFACE_INFERENCE_URI =
            "https://router.huggingface.co/hf-inference/models/{model}";
    public static final String POLLINATIONS_IMAGE_URI_TEMPLATE =
            "https://image.pollinations.ai/prompt/%s?width=%d&height=%d&nologo=true";

    public static final List<String> HUGGING_FACE_FALLBACK_MODELS = List.of(
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-3.5-medium",
            "stabilityai/sdxl-turbo",
            "Lykon/dreamshaper-xl-v2-turbo",
            "SG161222/RealVisXL_V4.0",
            "runwayml/stable-diffusion-v1-5"
    );

    public static final List<String> POLLINATIONS_FALLBACK_MODELS = List.of(
            "flux",
            "sana",
            "seedream",
            "gptimage",
            "zimage"
    );

    public static final byte PNG_SIGNATURE_BYTE = (byte) 0x89;
    public static final byte JPEG_SIGNATURE_BYTE_1 = (byte) 0xFF;
    public static final byte JPEG_SIGNATURE_BYTE_2 = (byte) 0xD8;
}
