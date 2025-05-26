package com.example.springaidemo.entity;

import lombok.Data;

@Data
public class QuestionRequest {
      private String question;
      private String sessionId;
}
