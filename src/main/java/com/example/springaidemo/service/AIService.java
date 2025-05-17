package com.example.springaidemo.service;

import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

@Service
public class AIService {

    private final ChatClient chatClient;

    public AIService(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    public String generateResponse(String userInput) {
        Prompt prompt = new Prompt(new UserMessage(userInput));
        return chatClient.call(prompt).getResult().getOutput().getContent();
    }
} 