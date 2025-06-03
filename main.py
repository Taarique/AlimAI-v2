# main.py

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# Import functions and configurations from your modules
import config # Import the whole config module
import gemini_integration # Import the whole gemini_integration module

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Command Handlers ---

async def start(update: Update, context):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name

    # Set default language and models for new users if not already set
    if user_id not in gemini_integration.user_data:
        gemini_integration.set_user_language(user_id, config.DEFAULT_LANGUAGE)
        gemini_integration.set_user_text_model(user_id, config.DEFAULT_TEXT_MODEL)
        gemini_integration.set_user_image_model(user_id, config.DEFAULT_IMAGE_MODEL)
        gemini_integration.reset_conversation(user_id) # Initialize chat sessions

    user_lang = gemini_integration.get_user_language(user_id)
    welcome_message = config.MESSAGES[user_lang]["start_welcome"].format(user_name=user_name)

    # Create inline keyboard for user guideline and about us
    keyboard = [
        [
            InlineKeyboardButton(config.MESSAGES[user_lang]["user_guideline_button"], callback_data='show_guideline'),
            InlineKeyboardButton(config.MESSAGES[user_lang]["about_us_button"], callback_data='show_about_us')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(welcome_message, parse_mode='MarkdownV2')
    await update.message.reply_text(config.MESSAGES[user_lang]["start_buttons_prompt"], reply_markup=reply_markup, parse_mode='MarkdownV2')


async def reset(update: Update, context):
    user_id = update.effective_user.id
    user_lang = gemini_integration.get_user_language(user_id)
    gemini_integration.reset_conversation(user_id)
    await update.message.reply_text(config.MESSAGES[user_lang]["reset_success"], parse_mode='MarkdownV2')

async def language_command(update: Update, context):
    user_id = update.effective_user.id
    user_lang = gemini_integration.get_user_language(user_id)

    keyboard = [
        [
            InlineKeyboardButton("English 🇬🇧", callback_data='set_lang_en'),
            InlineKeyboardButton("বাংলা 🇧🇩", callback_data='set_lang_bn'),
            InlineKeyboardButton("العربية 🇸🇦", callback_data='set_lang_ar')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(config.MESSAGES[user_lang]["language_prompt"], reply_markup=reply_markup, parse_mode='MarkdownV2')

async def model_command(update: Update, context):
    user_id = update.effective_user.id
    user_lang = gemini_integration.get_user_language(user_id)

    keyboard = []
    for key, model_info in config.AVAILABLE_TEXT_MODELS.items():
        description = model_info.get(f"description_{user_lang}", "") # Use localized description
        selected_indicator = " ✨" if key == gemini_integration.get_user_text_model(user_id) else ""
        # Check if API key is set for the model before allowing selection
        api_key_env_var = model_info.get("api_key_env")
        if not api_key_env_var or not os.getenv(api_key_env_var):
            status_indicator = " (API Key Missing ⚠️)"
            # Skip if API key is missing? Or just show a warning?
            # For now, we'll show it but with a warning.
        else:
            status_indicator = ""

        keyboard.append([InlineKeyboardButton(f"{model_info['name']}{selected_indicator}{status_indicator} - {description}", callback_data=f'set_model_{key}')])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(config.MESSAGES[user_lang]["model_prompt"], reply_markup=reply_markup, parse_mode='MarkdownV2')

async def image_command(update: Update, context):
    user_id = update.effective_user.id
    user_lang = gemini_integration.get_user_language(user_id)

    keyboard = []
    for key, model_info in config.AVAILABLE_IMAGE_MODELS.items():
        description = model_info.get(f"description_{user_lang}", "") # Use localized description
        selected_indicator = " ✨" if key == gemini_integration.get_user_image_model(user_id) else ""
        
        api_key_env_var = model_info.get("api_key_env")
        if api_key_env_var and not os.getenv(api_key_env_var):
            status_indicator = " (API Key Missing ⚠️)"
        else:
            status_indicator = ""

        keyboard.append([InlineKeyboardButton(f"{model_info['name']}{selected_indicator}{status_indicator} - {description}", callback_data=f'set_image_model_{key}')])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(config.MESSAGES[user_lang]["image_model_prompt"], reply_markup=reply_markup, parse_mode='MarkdownV2')

async def generate_image_command(update: Update, context):
    user_id = update.effective_user.id
    user_lang = gemini_integration.get_user_language(user_id)

    prompt = " ".join(context.args) # Get the description after /img

    if not prompt:
        await update.message.reply_text(config.MESSAGES[user_lang]["no_image_description"], parse_mode='MarkdownV2')
        return

    # Send typing/uploading photo action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
    await update.message.reply_text(config.MESSAGES[user_lang]["image_generation_start"])

    image_url = await gemini_integration.generate_image(prompt, user_id)

    if image_url and image_url.startswith("http"):
        try:
            await update.message.reply_photo(photo=image_url, caption=config.MESSAGES[user_lang]["image_generation_success"], parse_mode='MarkdownV2', reply_to_message_id=update.message.message_id)
        except Exception as e:
            logger.error(f"Error sending photo to Telegram for user {user_id}: {e}")
            await update.message.reply_text(config.MESSAGES[user_lang]["image_generation_failed"], parse_mode='MarkdownV2', reply_to_message_id=update.message.message_id)
    else:
        await update.message.reply_text(config.MESSAGES[user_lang]["image_generation_failed"], parse_mode='MarkdownV2', reply_to_message_id=update.message.message_id)


# --- Message Handler (for /ask and replies in groups, direct in private) ---

async def handle_message(update: Update, context):
    user_id = update.effective_user.id
    chat_type = update.effective_chat.type
    user_input = update.message.text
    user_lang = gemini_integration.get_user_language(user_id)

    if user_input is None: # Ignore non-text messages
        return

    is_reply_to_bot = False
    if update.message.reply_to_message:
        if update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.id == context.bot.id:
            is_reply_to_bot = True

    # Pre-check for non-Islamic keywords (simple filtering)
    # These keywords are case-insensitive. Add more as needed.
    non_islamic_keywords = ["politics", "entertainment", "sports", "recipe", "music", "movie", "stock", "nude", "porn", "sex",
                            "রাজনীতি", "বিনোদন", "খেলাধুলা", "রান্না", "গান", "চলচ্চিত্র", "শেয়ার", "নগ্ন", "পর্ন", "যৌন"]
    
    # Check if the user's input directly contains any problematic keywords
    if any(keyword in user_input.lower() for keyword in non_islamic_keywords):
        await update.message.reply_text(config.MESSAGES[user_lang]["only_islamic"], parse_mode='MarkdownV2', reply_to_message_id=update.message.message_id)
        return

    if chat_type == 'private':
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        response = await gemini_integration.get_ai_response(user_input, user_id)
        await update.message.reply_text(response, parse_mode='MarkdownV2')

    elif chat_type in ['group', 'supergroup']:
        # Handle /ask command
        if user_input.lower().startswith('/ask'):
            prompt = user_input[len('/ask'):].strip()
            if not prompt:
                await update.message.reply_text(config.MESSAGES[user_lang]["ask_command_usage"], parse_mode='MarkdownV2', reply_to_message_id=update.message.message_id)
                return
            
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            response = await gemini_integration.get_ai_response(prompt, user_id)
            await update.message.reply_text(response, parse_mode='MarkdownV2', reply_to_message_id=update.message.message_id)

        # Handle replies to the bot
        elif is_reply_to_bot:
            prompt = user_input
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            response = await gemini_integration.get_ai_response(prompt, user_id)
            await update.message.reply_text(response, parse_mode='MarkdownV2', reply_to_message_id=update.message.message_id)

        else:
            return # Ignore message if not a command or reply to bot in groups

# --- Callback Query Handlers ---

async def button_callback(update: Update, context):
    query = update.callback_query
    await query.answer()

    user_id = query.effective_user.id
    user_lang = gemini_integration.get_user_language(user_id) # Get current user language for messages

    if query.data.startswith('set_lang_'):
        lang_code = query.data.replace('set_lang_', '')
        
        lang_name_map = {"en": "English 🇬🇧", "bn": "বাংলা 🇧🇩", "ar": "العربية 🇸🇦"}
        gemini_integration.set_user_language(user_id, lang_code)
        
        confirmation_message = config.MESSAGES[lang_code]["lang_set_to"].format(lang_name=lang_name_map.get(lang_code, lang_code))
        await query.edit_message_text(text=confirmation_message, parse_mode='MarkdownV2')
        gemini_integration.reset_conversation(user_id) # Reset all text chat sessions after language change

    elif query.data.startswith('set_model_'): # For text models
        model_key_selected = query.data.replace('set_model_', '')
        # Check if API key is available before setting the model
        model_info = config.AVAILABLE_TEXT_MODELS.get(model_key_selected)
        if model_info and model_info.get("api_key_env") and not os.getenv(model_info["api_key_env"]):
            await query.edit_message_text(text=gemini_integration.escape_markdown_v2(f"⚠️ API key for {model_info['name']} is missing. Please set it in your environment variables."), parse_mode='MarkdownV2')
            return

        gemini_integration.set_user_text_model(user_id, model_key_selected)
        
        model_name_display = config.AVAILABLE_TEXT_MODELS.get(model_key_selected, {}).get("name", model_key_selected.upper())
        confirmation_message = config.MESSAGES[user_lang]["model_set_to"].format(model_name=model_name_display)
        await query.edit_message_text(text=confirmation_message, parse_mode='MarkdownV2')
        gemini_integration.reset_conversation(user_id, model_key_selected) # Reset specific chat session

    elif query.data.startswith('set_image_model_'): # For image models
        image_model_key_selected = query.data.replace('set_image_model_', '')
        # Check if API key is available for DeepAI
        model_info = config.AVAILABLE_IMAGE_MODELS.get(image_model_key_selected)
        if image_model_key_selected == "deepai" and model_info and model_info.get("api_key_env") and not os.getenv(model_info["api_key_env"]):
            await query.edit_message_text(text=gemini_integration.escape_markdown_v2(f"⚠️ API key for {model_info['name']} is missing. Please set it in your environment variables."), parse_mode='MarkdownV2')
            return

        gemini_integration.set_user_image_model(user_id, image_model_key_selected)

        model_name_display = config.AVAILABLE_IMAGE_MODELS.get(image_model_key_selected, {}).get("name", image_model_key_selected.upper())
        confirmation_message = config.MESSAGES[user_lang]["image_model_set_to"].format(model_name=model_name_display)
        await query.edit_message_text(text=confirmation_message, parse_mode='MarkdownV2')

    elif query.data == 'show_guideline':
        title = config.MESSAGES[user_lang]["guideline_title"]
        content = config.MESSAGES[user_lang]["guideline_content"]
        full_message = f"{title}\n\n{content}"
        await query.edit_message_text(text=full_message, parse_mode='MarkdownV2')

    elif query.data == 'show_about_us':
        title = config.MESSAGES[user_lang]["about_us_title"]
        content = config.MESSAGES[user_lang]["about_us_content"]
        full_message = f"{title}\n\n{content}"
        await query.edit_message_text(text=full_message, parse_mode='MarkdownV2')

    else:
        await query.edit_message_text(text=config.MESSAGES[user_lang]["unknown_action"], parse_mode='MarkdownV2')


# --- Main Function ---

def main():
    """Start the bot."""
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("restart", reset))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CommandHandler("img", generate_image_command)) # Works in both private and groups

    # Register callback query handler for all inline buttons
    app.add_handler(CallbackQueryHandler(button_callback, pattern='^set_(lang|model|image_model)_|^show_guideline$|^show_about_us$'))

    # Register message handler for text messages (excluding specific commands handled above)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)) # Keep ~filters.COMMAND here to let CommandHandlers take precedence

    logger.info("🤖 AlimAI Bot started...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()