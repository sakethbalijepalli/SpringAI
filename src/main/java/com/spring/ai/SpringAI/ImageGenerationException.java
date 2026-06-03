package com.spring.ai.SpringAI;

/**
 * Thrown when all image generation providers/models fail to produce an image.
 */
public class ImageGenerationException extends RuntimeException {

    public ImageGenerationException(String message, Throwable cause) {
        super(message, cause);
    }
}
