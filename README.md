# Spring AI Demo

This project demonstrates the usage of Spring AI with OpenAI integration. It provides a simple REST API to interact with AI models.

## Prerequisites

- Java 17 or higher
- Maven
- OpenAI API key

## Setup

1. Clone the repository
2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Building the Project

```bash
mvn clean install
```

## Running the Application

```bash
mvn spring-boot:run
```

The application will start on port 8080.

## Usage

To interact with the AI, send a POST request to the chat endpoint:

```bash
curl -X POST -H "Content-Type: text/plain" -d "What is Spring AI?" http://localhost:8080/api/ai/chat
```

## Project Structure

- `src/main/java/com/example/springaidemo/SpringAiDemoApplication.java`: Main application class
- `src/main/java/com/example/springaidemo/service/AIService.java`: Service handling AI interactions
- `src/main/java/com/example/springaidemo/controller/AIController.java`: REST controller
- `src/main/resources/application.properties`: Application configuration

## Dependencies

- Spring Boot 3.2.3
- Spring AI 0.8.1
- Spring Web