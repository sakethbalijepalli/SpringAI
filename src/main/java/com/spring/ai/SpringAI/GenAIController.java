package com.spring.ai.SpringAI;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClientResponseException;

import java.util.Map;

@RestController
public class GenAIController {

    private final ChatService chatService;
    private final ImageGenerationService imageGenerationService;

    public GenAIController(ChatService chatService, ImageGenerationService imageGenerationService) {
        this.chatService = chatService;
        this.imageGenerationService = imageGenerationService;
    }

    @GetMapping("/query-ai")
    public String getResponse(@RequestParam String prompt) {
        return chatService.getResponse(prompt);
    }

    @GetMapping("/query-ai-flash")
    public String getResponseOptions(@RequestParam String prompt) {
        return chatService.getResponseOptions(prompt);
    }

    @GetMapping(value = "/generate-image")
    public ResponseEntity<byte[]> generateImage(@RequestParam String prompt) {
        ImageGenerationService.GeneratedImage image = imageGenerationService.generate(prompt);
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType(image.contentType()))
                .body(image.bytes());
    }

    @ExceptionHandler(ChatException.class)
    public ResponseEntity<Map<String, String>> handleChatFailure(ChatException ex) {
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(Map.of("error", ex.getMessage()));
    }

    @ExceptionHandler(ImageGenerationException.class)
    public ResponseEntity<Map<String, String>> handleImageGeneration(ImageGenerationException ex) {
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(Map.of("error", ex.getMessage()));
    }

    @ExceptionHandler(RestClientResponseException.class)
    public ResponseEntity<Map<String, String>> handleUpstream(RestClientResponseException ex) {
        HttpStatus status = HttpStatus.resolve(ex.getStatusCode().value());
        if (status == null) {
            status = HttpStatus.BAD_GATEWAY;
        }
        return ResponseEntity.status(status)
                .body(Map.of("error", ex.getResponseBodyAsString().isBlank()
                        ? ex.getStatusText()
                        : ex.getResponseBodyAsString()));
    }
}
