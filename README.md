# Gradio Chat UI for Ollama

This project provides a simple and intuitive web-based chat interface for interacting with local language models hosted by **Ollama**. It leverages the Gradio library to create the UI and supports both text-only and multi-modal (text + image) conversations.

## Features

-   **Direct Ollama Integration**: Connects directly to your local Ollama instance.
    
-   **Model Selection**: Automatically discovers and lists all available Ollama models in a dropdown menu.
    
-   **Multi-modal Conversations**: Supports image uploads for visual question answering with compatible models (e.g., `qwen2.5vl`, `llava`).
    
-   **Custom System Prompts**: Easily modify the model's behavior by setting a custom system prompt directly in the UI.
    
-   **Advanced Parameter Tuning**: Adjust key generation parameters like `temperature`, `top_p`, `top_k`, and context window size (`num_ctx`) through an "Advanced Settings" panel.
    
-   **Clean & Simple Interface**: Built with Gradio for a user-friendly and responsive experience.
    

## Requirements

-   **Python 3.8+**
    
-   **Ollama**: You must have Ollama installed and running. See the official [Ollama website](https://ollama.com/ "null") for installation instructions.
    
-   At least one model pulled (e.g., `ollama pull qwen2.5vl:7b`).
    

### Dependencies

You can install the required Python packages using `pip`:

```
pip install gradio requests pillow langchain-core langchain-ollama

```

Or, run `pip install -r requirements.txt`:

```
gradio
requests
Pillow
langchain-core
langchain-ollama

```

## Setup & Installation

1.  **Clone the Repository**:
    
    ```
    git clone <your-repo-url>
    cd <your-repo-directory>
    
    ```
    
    (Or, simply save the Python script to a file, e.g., `app.py`).
    
2.  **Install Ollama & Models**:
    
    -   Download and install Ollama from [ollama.com](https://ollama.com/ "null").
        
    -   Run the Ollama application.
        
    -   Pull a model via the command line. For multi-modal support, a `vl` model is recommended:
        
        ```
        ollama pull qwen2.5vl:7b
        
        ```
        
    -   For a powerful text-only model, you could pull:
        
        ```
        ollama pull llama3
        
        ```
        
3.  **Install Python Dependencies**:
    
    ```
    pip install -r requirements.txt
    
    ```
    
4.  **(Optional) Configure Ollama Host**: The script defaults to connecting to `http://localhost:11434`. If your Ollama instance is running on a different host or port, set the `OLLAMA_HOST` environment variable:
    
    ```
    export OLLAMA_HOST="http://your-ollama-ip:11434"
    
    ```    

## How to Run

Execute the Python script from your terminal:

```
python main.py  # or the name you gave the script

```

Gradio will provide a local URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser to start using the chat interface.

### Using the Interface

1.  **Select a Model**: Choose from the list of available models you've pulled in Ollama.
    
2.  **Set the System Prompt**: (Optional) Modify the default system prompt to guide the model's personality or response style.
    
3.  **Adjust Advanced Settings**: (Optional) Expand the "Advanced Settings" section to fine-tune the model's generation parameters.
    
4.  **Start Chatting**: Type your message in the input box and press Enter or click "Send".
    
5.  **Upload an Image**: (Optional) Click the image box to upload a picture. The next message you send will be associated with that image.
    
6.  **Clear Chat**: Click the "Clear Chat" button to reset the conversation.
    

## Code Overview

-   **`get_ollama_models(ollama_host)`**: Fetches the list of locally available models from the Ollama API.
    
-   **`pil_to_base64(img, fmt)`**: A utility to convert uploaded images into the base64 format required for the API request.
    
-   **`build_chat_history(history)`**: Transforms the Gradio chat history format into a list of LangChain `HumanMessage` and `AIMessage` objects.
    
-   **`respond(...)`**: The core function that is triggered on user input. It assembles the final prompt (including system prompt, history, user message, and image), initializes `ChatOllama` with the selected parameters, and invokes the model to get a response.
    
-   **`gr.Blocks() as demo`**: This section defines the entire Gradio user interface, from the layout of components to their event listeners (`.click()`, `.submit()`).
