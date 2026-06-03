package com.spring.ai.SpringAI;

/**
 * Thrown when all chat providers/models fail to produce a response.
 */
public class ChatException extends RuntimeException {

    public ChatException(String message, Throwable cause) {
        super(message, cause);
    }
}
