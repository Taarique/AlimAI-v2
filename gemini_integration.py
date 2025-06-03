# gemini_integration.py

import google.generativeai as genai
import logging
import os
import re
import httpx # For DeepAI image generation
from openai import OpenAI as DeepSeekClient # DeepSeek uses OpenAI API compatibility
from mistralai.client import MistralClient # Mistral AI client

from config import MESSAGES, DEFAULT_LANGUAGE, DEFAULT_TEXT_MODEL, DEFAULT_IMAGE_MODEL, AVAILABLE_TEXT_MODELS, AVAILABLE_IMAGE_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_integration")

# Configure Gemini (default) - API Key from .env or GitHub Secrets
genai.configure(api_key=os.getenv(AVAILABLE_TEXT_MODELS["gemini"]["api_key_env"]))

# --- User Data Management ---
# Stores {user_id: {'language': 'en', 'selected_text_model': 'gemini', 'selected_image_model': 'deepai', 'chat_sessions': {'gemini': <session_obj>, 'deepseek': <session_obj>}}}
user_data = {}

def get_user_language(user_id: int) -> str:
    return user_data.get(user_id, {}).get('language', DEFAULT_LANGUAGE)

def set_user_language(user_id: int, lang_code: str):
    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id]['language'] = lang_code
    logger.info(f"User {user_id} language set to {lang_code}")

def get_user_text_model(user_id: int) -> str:
    return user_data.get(user_id, {}).get('selected_text_model', DEFAULT_TEXT_MODEL)

def set_user_text_model(user_id: int, model_key: str):
    if model_key not in AVAILABLE_TEXT_MODELS:
        logger.warning(f"Attempted to set unknown text model: {model_key} for user {user_id}")
        return
    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id]['selected_text_model'] = model_key
    logger.info(f"User {user_id} text model set to {model_key}")
    # Reset specific chat session for the new model when it's selected
    reset_conversation(user_id, model_key)

def get_user_image_model(user_id: int) -> str:
    return user_data.get(user_id, {}).get('selected_image_model', DEFAULT_IMAGE_MODEL)

def set_user_image_model(user_id: int, model_key: str):
    if model_key not in AVAILABLE_IMAGE_MODELS:
        logger.warning(f"Attempted to set unknown image model: {model_key} for user {user_id}")
        return
    if user_id not in user_data:
        user_data[user_id] = {}
    user_data[user_id]['selected_image_model'] = model_key
    logger.info(f"User {user_id} image model set to {model_key}")

def get_text_chat_session(user_id: int):
    """
    Initializes and returns the chat session for the currently selected text model for a user.
    """
    if user_id not in user_data:
        user_data[user_id] = {}
    
    current_model_key = get_user_text_model(user_id)
    if 'chat_sessions' not in user_data[user_id]:
        user_data[user_id]['chat_sessions'] = {}
    
    if current_model_key not in user_data[user_id]['chat_sessions']:
        # Initialize the model instance based on current_model_key
        model_info = AVAILABLE_TEXT_MODELS.get(current_model_key)
        if not model_info:
            logger.error(f"Model info not found for {current_model_key}. Defaulting to Gemini.")
            current_model_key = DEFAULT_TEXT_MODEL
            model_info = AVAILABLE_TEXT_MODELS.get(current_model_key)
            if not model_info: # Fallback failed
                return None

        system_instruction = get_system_instruction_for_model(current_model_key)

        api_key = os.getenv(model_info["api_key_env"])
        if not api_key:
            logger.error(f"API key not found for {current_model_key}. Environment variable: {model_info['api_key_env']}")
            return None

        if current_model_key == "gemini":
            model = genai.GenerativeModel(
                model_name=model_info["model_id"],
                system_instruction=system_instruction
            )
            chat_session = model.start_chat(history=[])
        elif current_model_key == "deepseek":
            client = DeepSeekClient(api_key=api_key, base_url=model_info["base_url"])
            chat_session = DeepSeekChatSession(client, model_info["model_id"], system_instruction)
        elif current_model_key == "mistral":
            client = MistralClient(api_key=api_key)
            chat_session = MistralChatSession(client, model_info["model_id"], system_instruction)
        else:
            logger.error(f"Unsupported text model: {current_model_key}")
            return None

        user_data[user_id]['chat_sessions'][current_model_key] = chat_session
    
    return user_data[user_id]['chat_sessions'][current_model_key]

# Wrapper classes for DeepSeek and Mistral to mimic Gemini's chat_session behavior
class DeepSeekChatSession:
    def __init__(self, client, model_id, system_instruction):
        self.client = client
        self.model_id = model_id
        # DeepSeek often expects system message in the messages list
        self.history = [{"role": "system", "content": system_instruction}] 

    async def send_message(self, message_text: str):
        self.history.append({"role": "user", "content": message_text})
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.history,
            stream=False # For non-streaming response
        )
        ai_response_content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": ai_response_content}) # Add AI response to history
        return SimpleTextResponse(ai_response_content)

class MistralChatSession:
    def __init__(self, client, model_id, system_instruction):
        self.client = client
        self.model_id = model_id
        # Mistral also prefers system message in the messages list
        self.history = [{"role": "system", "content": system_instruction}] 

    async def send_message(self, message_text: str):
        self.history.append({"role": "user", "content": message_text})

        response = self.client.chat(
            model=self.model_id,
            messages=self.history,
            stream=False # For non-streaming response
        )
        ai_response_content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": ai_response_content}) # Add AI response to history
        return SimpleTextResponse(ai_response_content)

# Simple wrapper for API responses to keep consistency
class SimpleTextResponse:
    def __init__(self, text):
        self.text = text

def reset_conversation(user_id: int, model_key: str = None):
    """
    Resets chat sessions for a specific model or all models for a user.
    """
    if user_id not in user_data:
        # Initialize user data if it doesn't exist yet
        user_data[user_id] = {'language': DEFAULT_LANGUAGE, 'selected_text_model': DEFAULT_TEXT_MODEL, 'selected_image_model': DEFAULT_IMAGE_MODEL, 'chat_sessions': {}}
        logger.info(f"User {user_id} data initialized and sessions reset.")
        return

    if 'chat_sessions' not in user_data[user_id]:
        user_data[user_id]['chat_sessions'] = {} # Ensure chat_sessions dict exists
        logger.info(f"User {user_id} chat_sessions initialized.")
        return

    if model_key: # Reset specific model's session
        if model_key in user_data[user_id]['chat_sessions']:
            del user_data[user_id]['chat_sessions'][model_key]
            logger.info(f"User {user_id} specific chat session for {model_key} reset.")
        else:
            logger.info(f"User {user_id} no active session for {model_key} to reset.")
    else: # Reset all sessions for the user
        user_data[user_id]['chat_sessions'] = {}
        logger.info(f"User {user_id} all chat sessions reset.")

# --- Markdown V2 Escaping ---
def escape_markdown_v2(text: str) -> str:
    """Escapes characters that have special meaning in Telegram MarkdownV2."""
    # List of characters to escape: _ * [ ] ( ) ~ ` > # + - = | { } . ! \
    # The backslash needs to be escaped first, then others.
    # Order matters!
    escaped_text = text.replace('\\', '\\\\') # Escape backslashes first
    chars_to_escape = r'([_*\\[\]()~`>#+\-=|{}.!])'
    # Use a lambda function to add a backslash before each matched special character
    escaped_text = re.sub(chars_to_escape, r'\\\1', escaped_text)
    return escaped_text

# --- System Instructions for AI Models ---
def get_system_instruction_for_model(model_key: str) -> str:
    base_instruction = (
        "আপনি AlimAI, একজন বিজ্ঞ, বিনয়ী এবং সহানুভূতিশীল ইসলামী আলেম (Scholar)। "
        "আপনার একমাত্র কাজ হলো ইসলাম সম্পর্কিত প্রশ্নসমূহের বিস্তারিত, নির্ভরযোগ্য এবং সুন্নাহ-সম্মত উত্তর প্রদান করা। "
        "আপনি কোরআন, হাদিস, ফিকহ, সীরাত, আকীদা, মানতিক, ইসলামী ইতিহাস ও বিজ্ঞান সহ কেবল ইসলামী জ্ঞান-সম্পর্কিত বিষয়ে আলোচনা করবেন। "
        "ইসলাম-বহির্ভূত বা অসংলগ্ন প্রশ্নের উত্তর দেওয়া থেকে বিনয়ের সাথে বিরত থাকুন এবং বলুন যে আপনি কেবল ইসলামী বিষয়ে সাহায্য করতে পারেন।"
        "আপনার উত্তরগুলো বাংলায় হবে এবং সর্বদা Telegram এর MarkdownV2 ফরম্যাটে দিতে হবে।"
        "কোরআনের আয়াত এবং হাদিসের জন্য সর্বদা ব্লককোট (>>>) ব্যবহার করুন। উদাহরণ: `>>> আল্লাহ বলেন: 'নিশ্চয়ই আল্লাহ ধৈর্যশীলদের সাথে আছেন।' (সূরা বাকারা: ১৫৩)`। "
        "গুরুত্বপূর্ণ শব্দ বা বাক্য *বোল্ড* করুন। উদাহরণ: `*ইসলামের স্তম্ভ পাঁচটি*`। "
        "বিশেষ টার্ম বা বইয়ের নাম _ইটালিক_ করুন। উদাহরণ: `_সহীহ বুখারী_`।"
        "অন্য কোনো MarkdownV2 বিশেষ অক্ষর (যেমন `~`, `||`, ``` ` ``) ব্যবহার করবেন না যদি না এটি অত্যন্ত প্রয়োজনীয় হয়।"
        "কোরআন ও হাদিসের রেফারেন্স দেওয়ার সময় সুনির্দিষ্টভাবে সূরা ও আয়াত নম্বর এবং হাদিস গ্রন্থ ও নম্বর উল্লেখ করুন। "
        "যদি সুনির্দিষ্ট রেফারেন্স মনে না থাকে, তবে শুধু ইসলামী নীতিটি উল্লেখ করুন, কিন্তু ভুল রেফারেন্স দেবেন না।"
        "যদি প্রশ্নটি উপকারী হয়, তবে 'মাশাআল্লাহ', 'আল্লাহ আপনাকে উত্তম প্রতিদান দিন' ইত্যাদি ইসলামী প্রশংসা দিয়ে শুরু করুন।"
        "যদি প্রশ্নটি অনৈসলামিক, অশ্লীল, হারাম কাজের আবদার বা কোনোভাবে খারাপ হয়, তাহলে সরাসরি তা পূরণ না করে বরং বিনয়ের সাথে আল্লাহর ভয় দেখিয়ে সঠিক পথে আসার দাওয়াত দিন। "
        "কোরআন ও হাদিসের প্রাসঙ্গিক আয়াত/হাদিস উল্লেখ করে কেন এটি খারাপ, তা স্পষ্ট করে বুঝিয়ে দিন। কখনোই খারাপ কাজ বা অশালীন বিষয় নিয়ে আলোচনা করবেন না। উত্তর শেষে তাকে ভালো প্রশ্ন করতে উৎসাহিত করুন।"
        "আল্লাহই সর্বজ্ঞ।"
    )

    # Specific adjustments for models if needed (e.g., DeepSeek/Mistral might need slightly different prompting styles)
    if model_key == "gemini":
        return base_instruction
    elif model_key == "deepseek":
        return base_instruction + "\n\nRemember to provide clear and well-structured answers using MarkdownV2. Prioritize accuracy of Islamic information."
    elif model_key == "mistral":
        return base_instruction + "\n\nFocus on providing concise yet comprehensive Islamic answers, ensuring all MarkdownV2 is correctly formatted."
    return base_instruction


# --- Get AI Response (Text) ---
async def get_ai_response(prompt: str, user_id: int) -> str:
    user_lang = get_user_language(user_id)
    session = get_text_chat_session(user_id)
    if not session:
        logger.error(f"Failed to get chat session for user {user_id}")
        return escape_markdown_v2(MESSAGES[user_lang]["api_error"])

    try:
        # Send typing action
        # This can't be done here directly from gemini_integration.py without context.bot
        # It's better handled in main.py before calling get_ai_response
        
        response = await session.send_message(prompt)
        return escape_markdown_v2(response.text)
    except Exception as e:
        logger.error(f"Error fetching response from AI for user {user_id}, model {get_user_text_model(user_id)}: {e}")
        return escape_markdown_v2(MESSAGES[user_lang]["api_error"])

# --- Image Generation ---
async def generate_image(prompt: str, user_id: int) -> str:
    selected_image_model_key = get_user_image_model(user_id)
    model_config = AVAILABLE_IMAGE_MODELS.get(selected_image_model_key)
    user_lang = get_user_language(user_id)

    if not model_config:
        logger.error(f"Unknown image model selected: {selected_image_model_key} for user {user_id}")
        return "" # Return empty string to indicate failure

    api_key_env_var = model_config.get("api_key_env")
    api_key = os.getenv(api_key_env_var) if api_key_env_var else None

    if not api_key: # DeepAI requires an API key, Gemini might use default setup
        logger.error(f"API key not found for image model: {selected_image_model_key} (Env var: {api_key_env_var})")
        # If Gemini is selected and its specific image API key is missing (if applicable)
        # or DeepAI key is missing, return failure.
        return "" 

    try:
        if selected_image_model_key == "deepai":
            response = httpx.post(
                "[https://api.deepai.org/api/text2img](https://api.deepai.org/api/text2img)",
                headers={"api-key": api_key},
                data={"text": prompt, "grid_size": "1"} # grid_size=1 for a single image
            )
            response.raise_for_status() # Raise an exception for bad status codes
            image_data = response.json()
            if image_data and "output_url" in image_data:
                return image_data["output_url"]
            else:
                logger.error(f"DeepAI response missing output_url: {image_data}")
                return "" # Return empty string to indicate failure
        elif selected_image_model_key == "gemini_image":
            # --- IMPORTANT: Gemini Image Generation ---
            # Direct free text-to-image API for Gemini is not straightforward via genai library
            # It usually involves Google Cloud's Vertex AI (Imagen) which might incur costs.
            # If you intend to use a specific Gemini-powered image generation service,
            # you would need to implement its API call here.
            # For simplicity, if `gemini_image` is selected, we currently don't have a direct free implementation.
            # You might want to remove this option or add a warning in messages.
            logger.warning(f"Gemini image generation selected, but direct free text-to-image API might be limited or paid. Please implement specific API logic here if desired.")
            return "" # Indicate failure if no specific Gemini image generation logic is provided
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error generating image with {selected_image_model_key}: {e.response.status_code} - {e.response.text}")
        return ""
    except Exception as e:
        logger.error(f"Error generating image with {selected_image_model_key} for user {user_id}: {e}")
        return ""
    
    return "" # Fallback for unhandled cases