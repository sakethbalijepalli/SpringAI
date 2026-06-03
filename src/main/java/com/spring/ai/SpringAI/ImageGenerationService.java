package com.spring.ai.SpringAI;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static com.spring.ai.SpringAI.ImageGenerationConstants.*;

/**
 * Orchestrates image generation across multiple providers with automatic
 * fallback. Each provider can itself rotate through a list of candidate models
 * before the orchestrator moves on to the next provider.
 */
@Service
public class ImageGenerationService {

    private static final Logger logger = LoggerFactory.getLogger(ImageGenerationService.class);

    /** Wrapper carrying raw image bytes together with the detected MIME type. */
    public record GeneratedImage(byte[] bytes, String contentType) {}

    private final RestClient httpClient;
    private final RestClient openRouterClient;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private final String provider;
    private final String openRouterModel;
    private final String googleModel;
    private final String googleApiKey;
    private final String huggingFaceApiKey;
    private final String huggingFaceModel;
    private final int pollinationsWidth;
    private final int pollinationsHeight;

    public ImageGenerationService(
            @Value("${app.image.provider:auto}") String provider,
            @Value("${spring.ai.openai.api-key:}") String openRouterApiKey,
            @Value("${spring.ai.openai.base-url:}") String openRouterBaseUrl,
            @Value("${spring.ai.openai.image.options.model:google/gemini-3.1-flash-image-preview}") String openRouterModel,
            @Value("${app.image.google-model:gemini-3-pro-image}") String googleModel,
            @Value("${spring.ai.google.genai.api-key:}") String googleApiKey,
            @Value("${app.image.huggingface.api-key:}") String huggingFaceApiKey,
            @Value("${app.image.huggingface.model:black-forest-labs/FLUX.1-schnell}") String huggingFaceModel,
            @Value("${app.image.pollinations.width:512}") int pollinationsWidth,
            @Value("${app.image.pollinations.height:512}") int pollinationsHeight) {

        this.provider           = provider;
        this.openRouterModel    = openRouterModel;
        this.googleModel        = googleModel;
        this.googleApiKey       = googleApiKey;
        this.huggingFaceApiKey  = huggingFaceApiKey;
        this.huggingFaceModel   = huggingFaceModel;
        this.pollinationsWidth  = pollinationsWidth;
        this.pollinationsHeight = pollinationsHeight;

        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(CONNECT_TIMEOUT);
        requestFactory.setReadTimeout(READ_TIMEOUT);
        this.httpClient = RestClient.builder().requestFactory(requestFactory).build();

        if (isNotBlank(openRouterBaseUrl) && isNotBlank(openRouterApiKey)) {
            String baseUrl = openRouterBaseUrl.trim();
            if (!baseUrl.endsWith("/v1") && !baseUrl.endsWith("/v1/")) {
                baseUrl = baseUrl.endsWith("/") ? baseUrl + "v1" : baseUrl + "/v1";
            }
            this.openRouterClient = RestClient.builder()
                    .baseUrl(baseUrl)
                    .defaultHeader("Authorization", "Bearer " + openRouterApiKey)
                    .build();
        } else {
            this.openRouterClient = null;
        }
    }

    /**
     * Generates an image for the given {@code prompt} by trying each configured
     * provider in order until one succeeds.
     *
     * @throws ImageGenerationException if every provider fails
     */
    public GeneratedImage generate(String prompt) {
        List<String> sequence   = buildProviderSequence();
        List<Throwable> errors  = new ArrayList<>();

        for (String p : sequence) {
            if (!isProviderConfigured(p)) {
                logger.info("Skipping provider '{}' — not fully configured.", p);
                continue;
            }
            try {
                logger.info("Attempting image generation via '{}'…", p);
                GeneratedImage image = dispatchToProvider(p, prompt);
                logger.info("Successfully generated image using '{}'.", p);
                return image;
            } catch (Exception ex) {
                logger.warn("Provider '{}' failed: {}", p, ex.getMessage());
                errors.add(ex);
            }
        }

        ImageGenerationException allFailed =
                new ImageGenerationException("All image generation providers failed.", null);
        errors.forEach(allFailed::addSuppressed);
        throw allFailed;
    }

    private GeneratedImage dispatchToProvider(String providerName, String prompt) {
        return switch (providerName) {
            case PROVIDER_POLLINATIONS -> generateViaPollinations(prompt);
            case PROVIDER_HUGGINGFACE  -> generateViaHuggingFace(prompt);
            case PROVIDER_GOOGLE       -> new GeneratedImage(generateViaGoogle(prompt), MediaType.IMAGE_PNG_VALUE);
            case PROVIDER_OPENROUTER   -> new GeneratedImage(generateViaOpenRouter(prompt), MediaType.IMAGE_PNG_VALUE);
            default -> throw new IllegalStateException("Unknown provider: " + providerName);
        };
    }

    private GeneratedImage generateViaPollinations(String prompt) {
        String encodedPrompt = URLEncoder.encode(prompt, StandardCharsets.UTF_8);

        // Try default (no model param) first, then each fallback model
        List<String> modelsToTry = new ArrayList<>();
        modelsToTry.add("");   // empty → default model
        modelsToTry.addAll(POLLINATIONS_FALLBACK_MODELS);

        List<Throwable> errors = new ArrayList<>();
        for (String model : modelsToTry) {
            String modelLabel = model.isEmpty() ? "default" : model;
            String url = POLLINATIONS_IMAGE_URI_TEMPLATE
                    .formatted(encodedPrompt, pollinationsWidth, pollinationsHeight);
            if (!model.isEmpty()) {
                url += "&model=" + model;
            }

            try {
                logger.info("Pollinations → trying model '{}'…", modelLabel);
                byte[] imageBytes = httpClient.get()
                        .uri(url)
                        .retrieve()
                        .body(byte[].class);

                validateImageBytes(imageBytes, "Pollinations");
                if (looksLikeJson(imageBytes)) {
                    throw new ImageGenerationException(parsePollinationsError(imageBytes), null);
                }
                logger.info("Pollinations → model '{}' succeeded.", modelLabel);
                return new GeneratedImage(imageBytes, detectContentType(imageBytes));
            } catch (RestClientResponseException ex) {
                String msg = "Pollinations model '%s' unavailable (%d): %s"
                        .formatted(modelLabel, ex.getStatusCode().value(), extractErrorMessage(ex));
                logger.warn(msg);
                errors.add(new ImageGenerationException(msg, ex));
            } catch (Exception ex) {
                logger.warn("Pollinations model '{}' failed: {}", modelLabel, ex.getMessage());
                errors.add(ex);
            }
        }

        throw aggregateException("All Pollinations models failed.", errors);
    }

    private GeneratedImage generateViaHuggingFace(String prompt) {
        List<String> modelsToTry = new ArrayList<>();
        if (isNotBlank(huggingFaceModel)) {
            modelsToTry.add(huggingFaceModel);
        }
        for (String m : HUGGING_FACE_FALLBACK_MODELS) {
            if (!modelsToTry.contains(m)) {
                modelsToTry.add(m);
            }
        }

        List<Throwable> errors = new ArrayList<>();
        for (String model : modelsToTry) {
            try {
                logger.info("Hugging Face → trying model '{}'…", model);
                byte[] imageBytes = httpClient.post()
                        .uri(HUGGINGFACE_INFERENCE_URI, model)
                        .header("Authorization", "Bearer " + huggingFaceApiKey)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body(Map.of("inputs", prompt))
                        .retrieve()
                        .body(byte[].class);

                validateImageBytes(imageBytes, "Hugging Face");
                if (looksLikeJson(imageBytes)) {
                    JsonNode error = parseJson(new String(imageBytes, StandardCharsets.UTF_8));
                    throw new ImageGenerationException(
                            error.path("error").asText(error.path("message").asText("Hugging Face request failed.")),
                            null);
                }
                logger.info("Hugging Face → model '{}' succeeded.", model);
                return new GeneratedImage(imageBytes, detectContentType(imageBytes));
            } catch (Exception ex) {
                logger.warn("Hugging Face model '{}' failed: {}", model, ex.getMessage());
                errors.add(ex);
            }
        }

        throw aggregateException("All Hugging Face models failed.", errors);
    }

    private byte[] generateViaGoogle(String prompt) {
        Map<String, Object> body = Map.of(
                "contents", new Object[]{
                        Map.of("parts", new Object[]{Map.of("text", prompt)})
                },
                "generationConfig", Map.of("responseModalities", new String[]{"IMAGE", "TEXT"}));

        String response = httpClient.post()
                .uri(GOOGLE_GENERATE_URI, googleModel, googleApiKey)
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(String.class);

        return extractGoogleImageBytes(parseJson(response));
    }

    private byte[] generateViaOpenRouter(String prompt) {
        if (openRouterClient == null) {
            throw new ImageGenerationException("OpenRouter client is not initialised.", null);
        }
        Map<String, Object> body = Map.of(
                "model", openRouterModel,
                "messages", new Object[]{Map.of("role", "user", "content", prompt)},
                "modalities", new String[]{"image", "text"});

        String response = openRouterClient.post()
                .uri(OPENROUTER_COMPLETIONS_PATH)
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(String.class);

        return extractOpenRouterImageBytes(parseJson(response));
    }

    /**
     * Builds the ordered list of providers to attempt. If a specific provider
     * was configured (not "auto"), it leads the sequence.
     */
    private List<String> buildProviderSequence() {
        List<String> sequence = new ArrayList<>();
        String primary = (provider != null) ? provider.trim().toLowerCase() : "";

        if (!primary.isEmpty() && !primary.equals(PROVIDER_AUTO)) {
            sequence.add(primary);
        }
        for (String p : DEFAULT_PROVIDER_ORDER) {
            if (!sequence.contains(p)) {
                sequence.add(p);
            }
        }
        return sequence;
    }

    private boolean isProviderConfigured(String p) {
        return switch (p) {
            case PROVIDER_GOOGLE      -> isNotBlank(googleApiKey);
            case PROVIDER_OPENROUTER  -> openRouterClient != null;
            case PROVIDER_HUGGINGFACE -> isNotBlank(huggingFaceApiKey);
            case PROVIDER_POLLINATIONS -> true;
            default -> false;
        };
    }

    private byte[] extractGoogleImageBytes(JsonNode root) {
        JsonNode parts = root.path("candidates").path(0).path("content").path("parts");
        if (!parts.isArray()) {
            throw new ImageGenerationException("Google response contained no image parts.", null);
        }
        for (JsonNode part : parts) {
            JsonNode inlineData = part.path("inlineData");
            if (inlineData.has("data")) {
                return Base64.getDecoder().decode(inlineData.path("data").asText());
            }
        }
        throw new ImageGenerationException("Google response contained no inline image data.", null);
    }

    private byte[] extractOpenRouterImageBytes(JsonNode root) {
        JsonNode images = root.path("choices").path(0).path("message").path("images");
        if (!images.isArray() || images.isEmpty()) {
            throw new ImageGenerationException("OpenRouter response contained no images.", null);
        }
        String dataUrl = images.path(0).path("image_url").path("url").asText(null);
        if (dataUrl == null || dataUrl.isBlank()) {
            throw new ImageGenerationException("OpenRouter image entry had no URL.", null);
        }
        return decodeDataUrl(dataUrl);
    }

    private void validateImageBytes(byte[] imageBytes, String providerLabel) {
        if (imageBytes == null || imageBytes.length == 0) {
            throw new ImageGenerationException(providerLabel + " returned an empty image.", null);
        }
    }

    private boolean looksLikeJson(byte[] bytes) {
        return bytes.length > 0 && bytes[0] == '{';
    }

    private String parsePollinationsError(byte[] bytes) {
        JsonNode body = parseJson(new String(bytes, StandardCharsets.UTF_8));
        if (body.has("error")) {
            return body.path("error").asText();
        }
        return "Pollinations rate limit reached — try huggingface or google provider.";
    }

    private String extractErrorMessage(RestClientResponseException ex) {
        try {
            JsonNode body = objectMapper.readTree(ex.getResponseBodyAsString());
            if (body.has("error")) {
                return body.get("error").isTextual()
                        ? body.path("error").asText()
                        : body.path("error").path("message").asText(ex.getStatusText());
            }
        } catch (Exception ignored) {
            // fall through to status text
        }
        return ex.getStatusText();
    }

    private String detectContentType(byte[] imageBytes) {
        if (imageBytes.length >= 8
                && imageBytes[0] == PNG_SIGNATURE_BYTE
                && imageBytes[1] == 'P'
                && imageBytes[2] == 'N'
                && imageBytes[3] == 'G') {
            return MediaType.IMAGE_PNG_VALUE;
        }
        if (imageBytes.length >= 2
                && imageBytes[0] == JPEG_SIGNATURE_BYTE_1
                && imageBytes[1] == JPEG_SIGNATURE_BYTE_2) {
            return MediaType.IMAGE_JPEG_VALUE;
        }
        return MediaType.APPLICATION_OCTET_STREAM_VALUE;
    }

    private byte[] decodeDataUrl(String dataUrl) {
        int comma = dataUrl.indexOf(',');
        String base64 = comma >= 0 ? dataUrl.substring(comma + 1) : dataUrl;
        return Base64.getDecoder().decode(base64);
    }

    private JsonNode parseJson(String response) {
        try {
            return objectMapper.readTree(response);
        } catch (Exception e) {
            throw new ImageGenerationException("Failed to parse image API response.", e);
        }
    }

    private ImageGenerationException aggregateException(String message, List<Throwable> causes) {
        ImageGenerationException ex = new ImageGenerationException(message, null);
        causes.forEach(ex::addSuppressed);
        return ex;
    }

    private static boolean isNotBlank(String value) {
        return value != null && !value.isBlank();
    }
}
