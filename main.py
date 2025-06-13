import os
import base64
from io import BytesIO
from typing import List, Tuple, Union, Optional

import gradio as gr
import requests
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

# The host URL for the Ollama API. It defaults to localhost.
# Can be overridden by setting the OLLAMA_HOST environment variable.
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# The default model to use if none is selected or available.
# This model is vision-capable (vl).
DEFAULT_MODEL_NAME: str = "qwen2.5vl:7b"

# A default instruction for the model on how to behave.
# This can be changed by the user in the UI.
DEFAULT_SYSTEM_PROMPT: str = "You are a helpful, respectful, and honest assistant."


# --- Default model parameters ---

# Controls the randomness of the model's output. Higher values (e.g., 0.8)
# make the output more creative and diverse, while lower values (e.g., 0.2)
# make it more deterministic and focused.
DEFAULT_TEMPERATURE: float = 0.5

# Nucleus sampling parameter. The model considers only the tokens whose
# cumulative probability mass is greater than top_p. A value of 0.9 means
# the model will consider the top 90% of the most likely tokens.
DEFAULT_TOP_P: float = 0.9

# Restricts the model's choices to the top 'k' most likely tokens at each step.
# A lower 'k' (e.g., 40) can make the output more coherent but less varied.
DEFAULT_TOP_K: int = 40

# The size of the context window (in tokens) that the model considers from
# the chat history when generating a new response.
DEFAULT_NUM_CTX: int = 2048


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_ollama_models(ollama_host: str) -> List[str]:
    """
    Get a list of available models from the Ollama API.

    Args:
        ollama_host (str): The URL of the Ollama server.

    Returns:
        List[str]: A list of model names (e.g., ['llama3', 'mistral']).
                   Returns a list with the default model name on failure.
    """
    try:
        # Send a GET request to the Ollama API's /api/tags endpoint.
        response = requests.get(f"{ollama_host}/api/tags")
        # Raise an exception if the request returned an HTTP error status.
        response.raise_for_status()
        # Parse the JSON response.
        models = response.json().get("models", [])
        # Extract just the name of each model from the list of model objects.
        return [model["name"] for model in models]
    except (requests.exceptions.RequestException, KeyError) as e:
        # Handle cases where the Ollama server is not running or the response is malformed.
        print(f"⚠️ Could not fetch models from Ollama: {e}")
        print(f"-> Please ensure Ollama is running and accessible at {ollama_host}")
        # Fallback to the default model name so the UI can still start.
        return [DEFAULT_MODEL_NAME]

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """
    Convert a PIL image to a base64-encoded string.

    Args:
        img (Image.Image): The PIL Image object to convert.
        fmt (str): The image format to save as (e.g., "PNG", "JPEG").

    Returns:
        str: The base64-encoded string representation of the image.
    """
    # Ensure image is in RGB format, as some models don't handle palettes ('P') or alpha channels ('RGBA').
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    
    # Create an in-memory binary stream.
    buf = BytesIO()
    # Save the image to the buffer in the specified format.
    img.save(buf, format=fmt)
    # Encode the binary data to base64 and return it as a string.
    return base64.b64encode(buf.getvalue()).decode()


def build_chat_history(
    history: List[Tuple[str, str]]
) -> List[Union[HumanMessage, AIMessage]]:
    """
    Transform a Gradio chat history list into a list of LangChain message objects.

    Args:
        history (List[Tuple[str, str]]): The Gradio chat history, where each
                                         tuple is an (user_message, ai_message) pair.

    Returns:
        List[Union[HumanMessage, AIMessage]]: A list of LangChain message objects.
    """
    messages: List[Union[HumanMessage, AIMessage]] = []
    # Iterate over each user/AI turn in the history.
    for user_msg, ai_msg in history:
        # Create a HumanMessage for the user's part of the turn.
        messages.append(HumanMessage(content=user_msg))
        # Create an AIMessage for the model's response.
        messages.append(AIMessage(content=ai_msg))
    return messages

# ---------------------------------------------------------------------
# Core chat routine
# ---------------------------------------------------------------------

def respond(
    user_message: str,
    history: List[Tuple[str, str]],
    image_path: Optional[str],
    model_name: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_ctx: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generate a response from the LLM, handle image input, and update the chat history.

    Args:
        user_message (str): The latest message from the user.
        history (List[Tuple[str, str]]): The current chat history.
        image_path (Optional[str]): The file path to an optional user-uploaded image.
        model_name (str): The name of the Ollama model to use.
        system_prompt (str): The system prompt to guide the model's behavior.
        temperature (float): The temperature for sampling.
        top_p (float): The top-p value for nucleus sampling.
        top_k (int): The top-k value for sampling.
        num_ctx (int): The context window size.

    Returns:
        A tuple containing two copies of the updated chat history. This is needed
        to update both the chatbot component and the internal state in Gradio.
    """
    # If no model is selected, return an error message.
    if not model_name:
        history.append((user_message, "**Error: No model selected.** Please choose a model from the dropdown menu."))
        return history, history

    # Initialize the ChatOllama model with the specified parameters from the UI.
    chat_model = ChatOllama(
        model=model_name,
        base_url=OLLAMA_HOST,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_ctx=num_ctx,
    )
    
    # Start building the list of messages to send to the model.
    # The first message is always the system prompt.
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = [
        SystemMessage(content=system_prompt)
    ]
    # Add the past conversation history.
    messages.extend(build_chat_history(history))

    # Handle optional image input.
    if image_path:
        try:
            # Open the image file from the provided path.
            pil_img = Image.open(image_path)
            # Convert the image to a base64 string.
            img_b64 = pil_to_base64(pil_img)
            # For multi-modal input, the content is a list of dictionaries.
            user_content = [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    # Format the image as a data URI for the model.
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ]
        except Exception as exc:
            # If image processing fails, fall back to just using the text message.
            print(f"⚠️ Image processing failed: {exc}")
            user_content = user_message
    else:
        # If no image is provided, the content is just the user's text message.
        user_content = user_message

    # Append the final user message (with or without an image) to the list.
    messages.append(HumanMessage(content=user_content))
    
    # Invoke the model with the complete message history.
    ai_resp: AIMessage = chat_model.invoke(messages)
    
    # Append the user's new message and the AI's response to the history.
    history.append((user_message, ai_resp.content))
    
    # Return the updated history twice for Gradio.
    return history, history


def clear() -> Tuple[List[Tuple[str, str]], str]:
    """
    Clears the chat history and the user input textbox.

    Returns:
        A tuple containing an empty list (for the chatbot) and an empty string
        (for the textbox).
    """
    return [], ""

# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------
# Create the main Gradio Blocks layout. `with gr.Blocks() as demo:` is the standard way to define a Gradio interface.
with gr.Blocks() as demo:
    # Add a title using Markdown.
    gr.Markdown("## 💬 Chat with Ollama LLMs (+ Image Support)")

    # Create a single row for the model selector and system prompt.
    with gr.Row():
        # Dropdown for selecting the Ollama model.
        model_selector = gr.Dropdown(
            label="Select Model",
            # Fetch the list of models from the Ollama server.
            choices=get_ollama_models(OLLAMA_HOST),
            # Set the initial value. If the default model is available, use it.
            # Otherwise, use the first available model.
            value=lambda: DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in get_ollama_models(OLLAMA_HOST) else (get_ollama_models(OLLAMA_HOST)[0] if get_ollama_models(OLLAMA_HOST) else None),
            scale=1,  # Relative width in the row.
        )        
        # Textbox for the system prompt.
        system_prompt_input = gr.Textbox(
            label="System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            scale=2, # Takes twice the width of the model selector.
            lines=2,
        )

    # Accordion for advanced model settings, initially closed.
    with gr.Accordion("Advanced Settings", open=False):
        # Slider for Temperature.
        temperature_slider = gr.Slider(
            minimum=0.0, maximum=2.0, step=0.05, value=DEFAULT_TEMPERATURE,
            label="Temperature", info="Controls randomness. Higher is more creative."
        )
        # Slider for Top-p.
        top_p_slider = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_TOP_P,
            label="Top-p (Nucleus Sampling)", info="Selects from tokens with a cumulative probability > p."
        )
        # Slider for Top-k.
        top_k_slider = gr.Slider(
            minimum=1, maximum=100, step=1, value=DEFAULT_TOP_K,
            label="Top-k", info="Restricts sampling to the top k most likely tokens."
        )
        # Number input for Context Window size.
        num_ctx_input = gr.Number(
            value=DEFAULT_NUM_CTX, label="Context Window (num_ctx)", precision=0,
            info="Number of tokens from history the model considers."
        )

    # The main chatbot display area.
    chatbot = gr.Chatbot(label="Chat History", height=450)

    # A row for the user input textbox and the send button.
    with gr.Row():
        # Textbox for user input.
        user_input = gr.Textbox(
            placeholder="Type your message...",
            show_label=False,
            scale=8, # Takes up most of the row width.
            autofocus=True,
        )
        # Send button.
        send_btn = gr.Button("Send", scale=1)

    # Image upload component.
    image_input = gr.Image(type="filepath", label="Optional image")
    # Button to clear the chat.
    clear_btn = gr.Button("Clear Chat")
    
    # List of all input components for the `respond` function.
    # The order must match the `respond` function's signature.
    param_inputs = [
        user_input, chatbot, image_input, model_selector,
        system_prompt_input, # The newly added system prompt input
        temperature_slider, top_p_slider, top_k_slider, num_ctx_input
    ]
    
    # Define the event listeners.
    
    # When the send button is clicked, call the `respond` function.
    # `inputs` maps the components in `param_inputs` to the function's arguments.
    # `outputs` maps the function's return values to the components to update.
    send_btn.click(respond, inputs=param_inputs, outputs=[chatbot, chatbot])
    
    # Allow submitting the message by pressing Enter in the textbox.
    user_input.submit(respond, inputs=param_inputs, outputs=[chatbot, chatbot])
    
    # When the clear button is clicked, call the `clear` function.
    # This will clear the chatbot display and the user input box.
    clear_btn.click(clear, outputs=[chatbot, user_input])

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
# The standard Python entry point.
if __name__ == "__main__":
    # Launch the Gradio application.
    demo.launch()