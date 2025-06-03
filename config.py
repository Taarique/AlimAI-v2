# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY") # Ensure this is also loaded

# --- Default Settings ---
DEFAULT_LANGUAGE = "en" # 'en' for English, 'bn' for Bangla, 'ar' for Arabic
DEFAULT_TEXT_MODEL = "gemini" # 'gemini', 'deepseek', 'mistral'
DEFAULT_IMAGE_MODEL = "deepai" # 'gemini_image', 'deepai'

# --- Available AI Models for Text Generation ---
AVAILABLE_TEXT_MODELS = {
    "gemini": {
        "name": "Google Gemini (Default)",
        "model_id": "gemini-1.5-flash", # Use gemini-1.5-flash for speed
        "api_key_env": "GEMINI_API_KEY",
        "description_en": "Fast and versatile AI model by Google.",
        "description_bn": "গুগলের দ্রুত এবং বহুমুখী এআই মডেল।",
        "description_ar": "نموذج ذكاء اصطناعي سريع ومتعدد الاستخدامات من جوجل."
    },
    "deepseek": {
        "name": "DeepSeek (Open)",
        "model_id": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "description_en": "Open and powerful AI model.",
        "description_bn": "একটি শক্তিশালী ওপেন এআই মডেল।",
        "description_ar": "نموذج ذكاء اصطناعي مفتوح وقوي."
    },
    "mistral": {
        "name": "Mistral (Open)",
        "model_id": "mistral-tiny", # or mistral-small, mistral-medium etc.
        "api_key_env": "MISTRAL_API_KEY",
        "description_en": "Efficient and high-performance open model.",
        "description_bn": "দক্ষ এবং উচ্চ-পারফরম্যান্স ওপেন মডেল।",
        "description_ar": "نموذج مفتوح فعال وعالي الأداء."
    }
}

# --- Available AI Models for Image Generation ---
AVAILABLE_IMAGE_MODELS = {
    "gemini_image": { # Placeholder for future direct Gemini image integration if available
        "name": "Google Gemini (Image)",
        "description_en": "Uses Google Gemini for image generation. (Requires separate setup or paid access, currently DeepAI is default fallback)",
        "description_bn": "ছবি তৈরির জন্য গুগল জেমিনি ব্যবহার করে। (আলাদা সেটআপ বা পেইড অ্যাক্সেস প্রয়োজন, বর্তমানে DeepAI ডিফল্ট ফলব্যাক।)",
        "description_ar": "يستخدم جوجل جيميني لتوليد الصور. (يتطلب إعدادًا منفصلاً أو وصولاً مدفوعًا، حاليًا DeepAI هو الخيار الافتراضي)."
    },
    "deepai": {
        "name": "DeepAI (Image)",
        "description_en": "Uses DeepAI for basic image generation. (Free, basic options)",
        "description_bn": "ছবি তৈরির জন্য ডিপএআই ব্যবহার করে। (ফ্রি, বেসিক অপশন)",
        "description_ar": "يستخدم ديب آي لتوليد الصور الأساسية. (مجاني، خيارات أساسية)",
        "api_key_env": "DEEPAI_API_KEY"
    }
}


# --- Multi-language Messages ---
MESSAGES = {
    "en": {
        "start_welcome": "Assalamu Alaikum wa Rahmatullahi, *{user_name}*!\n\nI am AlimAI, your digital companion for acquiring Islamic knowledge. I am ready to assist you with various Islamic topics including *Quran, Hadith, Fiqh, Mantiq, history, and science*.\n\nSend me your questions. You can restart or reset the conversation with /reset or /restart.",
        "start_buttons_prompt": "Explore AlimAI's features:",
        "user_guideline_button": "User Guideline 📖",
        "about_us_button": "About Us 💡",
        "reset_success": "♻️ Conversation has been reset. You can now ask a new question, InshaAllah.",
        "language_prompt": "🌐 Please choose your preferred language:",
        "only_islamic": "Dear brother/sister, I can only assist you with matters related to Islamic knowledge. InshaAllah, you can ask another Islamic question.",
        "lang_set_to": "Language set to *{lang_name}*.",
        "model_prompt": "🧠 Please choose your preferred AI model for text generation:",
        "model_set_to": "AI model set to *{model_name}*.",
        "image_model_prompt": "🖼️ Please choose an image generation model:",
        "image_generation_start": "Generating your image... Please wait a moment, InshaAllah.",
        "image_generation_success": "Here's your image, InshaAllah:",
        "image_generation_failed": "Sorry, I could not generate the image. Please try again or with a different description.",
        "img_command_usage": "Please use the command like: `/img [your image description]`",
        "no_image_description": "Please provide a description for the image you want to generate. Example: `/img a cat with a mosque background`",
        "ask_command_usage": "Please use `/ask` followed by your question. Example: `/ask What are the pillars of Islam?`",
        "api_error": "Sorry, I am facing an issue connecting to the AI service right now. Please try again later, InshaAllah.",
        "unknown_action": "Unknown action.",
        "guideline_title": "📖 *AlimAI User Guideline*",
        "guideline_content": (
            "*Welcome to AlimAI, your Islamic Knowledge Companion! Here's how to use me:*\n\n"
            "1\\. *Asking Questions:*\n"
            "   • *Private Chat: * Just type your question directly\\.\n"
            "   • *Group Chat: * Use `/ask` followed by your question \\(e\\.g\\., `/ask What are the pillars of Islam?`\\) or reply to one of my messages with your question\\.\n\n"
            "2\\. *Supported Topics:*\n"
            "   I strictly answer questions related to *Quran, Hadith, Fiqh, Seerah, Aqeedah, Mantiq, Islamic History, and Islamic Science*\\.\n"
            "   *I will politely decline questions outside these topics\\.*\n\n"
            "3\\. *Language Selection:*\n"
            "   Use the `/language` command to choose your preferred language: English, Bangla, or Arabic\\.\n\n"
            "4\\. *AI Model Selection:*\n"
            "   Use the `/model` command to select your preferred AI model for text generation \\(e\\.g\\., Gemini, DeepSeek\\)\\.\n\n"
            "5\\. *Image Generation:*\n"
            "   • Use the `/image` command to see available image generation models\\.\n"
            "   • To generate an image, use `/img` followed by a description \\(e\\.g\\., `/img a mosque in paradise`\\)\\.\n\n"
            "6\\. *Conversation Reset:*\n"
            "   Use `/reset` or `/restart` to clear the current conversation context and start fresh\\.\n\n"
            "7\\. *Islamic Guidance:*\n"
            "   I will always strive to answer in accordance with Islamic teachings and etiquettes\\. For inappropriate questions, I will provide Islamic guidance based on Quran and Hadith\\. Please respect these guidelines\\.\n\n"
            "*May Allah increase us in beneficial knowledge!*"
        ),
        "about_us_title": "💡 *About AlimAI & Our Vision*",
        "about_us_content": (
            "AlimAI is an innovative Telegram bot designed to be your trusted digital companion for exploring the vast ocean of Islamic knowledge\\. Our mission is to provide accurate, reliable, and spiritually uplifting answers based on the Holy Quran and authentic Sunnah\\.\n\n"
            "*Our Vision:*\n"
            "We envision a world where access to authentic Islamic knowledge is universally easy and immediate\\. AlimAI aims to bridge the gap between seekers of knowledge and the rich heritage of Islam, fostering a deeper understanding of our faith through cutting-edge AI technology, all while adhering strictly to Islamic principles and scholarly etiquette\\.\n\n"
            "We believe that technology, when guided by pure intentions, can be a powerful tool for spreading goodness and enlightenment\\. We are committed to continuous improvement, ensuring AlimAI remains a beneficial and reliable resource for Muslims worldwide\\.\n\n"
            "May Allah accept our humble efforts and make this project a source of immense benefit for the Ummah\\."
        )
    },
    "bn": {
        "start_welcome": "আসসালামু আলাইকুম ওয়া রাহমাতুল্লাহ, *{user_name}*!\n\nআমি AlimAI, আপনার ইসলামী জ্ঞান অর্জনের ডিজিটাল সঙ্গী। কোরআন, হাদিস, ফিকহ, মানতিক, ইতিহাস এবং বিজ্ঞান সহ ইসলামী জ্ঞানের বিভিন্ন বিষয়ে আপনাকে সহায়তা করতে আমি প্রস্তুত।\n\nআপনি যে প্রশ্ন করতে চান তা পাঠান। /restart বা /reset দিয়ে কনভারসেশন পুনরায় শুরু করতে পারেন।",
        "start_buttons_prompt": "AlimAI এর ফিচারগুলো এক্সপ্লোর করুন:",
        "user_guideline_button": "ব্যবহারকারী নির্দেশিকা 📖",
        "about_us_button": "আমাদের সম্পর্কে 💡",
        "reset_success": "♻️ কনভারসেশন রিসেট হয়েছে। এখন আপনি নতুন প্রশ্ন করতে পারেন ইনশাআল্লাহ।",
        "language_prompt": "🌐 অনুগ্রহ করে আপনার পছন্দের ভাষা নির্বাচন করুন:",
        "only_islamic": "প্রিয় ভাই/বোন, আমি কেবল ইসলামী জ্ঞান-সম্পর্কিত বিষয়ে আপনাকে সহায়তা করতে পারি। আল্লাহ চাহেত, আপনি অন্য কোনো ইসলামী প্রশ্ন করতে পারেন।",
        "lang_set_to": "ভাষা *{lang_name}* এ সেট করা হয়েছে।",
        "model_prompt": "🧠 টেক্সট জেনারেশনের জন্য আপনার পছন্দের AI মডেল নির্বাচন করুন:",
        "model_set_to": "AI মডেল *{model_name}* এ সেট করা হয়েছে।",
        "image_model_prompt": "🖼️ ছবি তৈরির জন্য একটি মডেল নির্বাচন করুন:",
        "image_generation_start": "আপনার ছবিটি তৈরি করা হচ্ছে... ইনশাআল্লাহ, একটু অপেক্ষা করুন।",
        "image_generation_success": "ইনশাআল্লাহ, আপনার ছবিটি এখানে:",
        "image_generation_failed": "দুঃখিত, ছবিটি তৈরি করতে পারিনি। অনুগ্রহ করে আবার চেষ্টা করুন অথবা ভিন্ন বর্ণনা দিন।",
        "img_command_usage": "অনুগ্রহ করে এভাবে কমান্ডটি ব্যবহার করুন: `/img [আপনার ছবির বর্ণনা]`",
        "no_image_description": "অনুগ্রহ করে যে ছবিটি তৈরি করতে চান তার একটি বর্ণনা দিন। উদাহরণ: `/img মসজিদের পটভূমিতে একটি বিড়াল`",
        "ask_command_usage": "অনুগ্রহ করে `/ask` এর পর আপনার প্রশ্নটি লিখুন। উদাহরণ: `/ask ইসলামের স্তম্ভ কয়টি?`",
        "api_error": "দুঃখিত, AI সার্ভিস এর সাথে সংযোগ করতে সমস্যা হচ্ছে। ইনশাআল্লাহ, কিছুক্ষণ পর আবার চেষ্টা করুন।",
        "unknown_action": "অজানা অ্যাকশন।",
        "guideline_title": "📖 *AlimAI ব্যবহারকারী নির্দেশিকা*",
        "guideline_content": (
            "*AlimAI, আপনার ইসলামী জ্ঞানের সঙ্গী, আপনাকে স্বাগতম! কিভাবে আমাকে ব্যবহার করবেন তা নিচে দেওয়া হলো:*\n\n"
            "১\\. *প্রশ্ন জিজ্ঞাসা:*\n"
            "   • *প্রাইভেট চ্যাট: * সরাসরি আপনার প্রশ্ন লিখুন\\.\n"
            "   • *গ্রুপ চ্যাট: * `/ask` কমান্ডের পর আপনার প্রশ্ন লিখুন \\(উদাহরণস্বরূপ, `/ask ইসলামের স্তম্ভ কয়টি?`\\) অথবা আমার পূর্বের কোনো মেসেজের রিপ্লাই দিয়ে প্রশ্ন করুন\\.\n\n"
            "২\\. *সমর্থিত বিষয়সমূহ:*\n"
            "   আমি শুধুমাত্র *কোরআন, হাদিস, ফিকহ, সীরাত, আকীদা, মানতিক, ইসলামী ইতিহাস এবং ইসলামী বিজ্ঞান* সম্পর্কিত প্রশ্নের উত্তর দিই\\.\n"
            "   *এই বিষয়গুলোর বাইরের প্রশ্নগুলো আমি বিনয়ের সাথে প্রত্যাখ্যান করব\\.*\n\n"
            "৩\\. *ভাষা নির্বাচন:*\n"
            "   `/language` কমান্ড ব্যবহার করে আপনার পছন্দের ভাষা নির্বাচন করুন: ইংরেজি, বাংলা অথবা আরবি\\.\n\n"
            "৪\\. *AI মডেল নির্বাচন:*\n"
            "   টেক্সট জেনারেশনের জন্য আপনার পছন্দের AI মডেল নির্বাচন করতে `/model` কমান্ড ব্যবহার করুন \\(যেমন, জেমিনি, ডিপসিক\\)\\.\n\n"
            "৫\\. *ছবি তৈরি:*\n"
            "   • ছবি তৈরির জন্য উপলব্ধ মডেলগুলো দেখতে `/image` কমান্ড ব্যবহার করুন\\.\n"
            "   • ছবি তৈরি করতে `/img` কমান্ডের পর ছবির বর্ণনা লিখুন \\(উদাহরণস্বরূপ, `/img জান্নাতে একটি মসজিদ`\\)\\.\n\n"
            "৬\\. *কথোপকথন রিসেট:*\n"
            "   কথোপকথনের প্রসঙ্গ পরিষ্কার করতে এবং নতুন করে শুরু করতে `/reset` অথবা `/restart` ব্যবহার করুন\\.\n\n"
            "৭\\. *ইসলামী দিকনির্দেশনা:*\n"
            "   আমি সর্বদা ইসলামী শিক্ষা ও আদব অনুযায়ী উত্তর দিতে সচেষ্ট থাকব\\. অনুপযুক্ত প্রশ্নের জন্য, আমি কোরআন ও হাদিসের ভিত্তিতে ইসলামী দিকনির্দেশনা প্রদান করব\\. অনুগ্রহ করে এই নির্দেশিকাগুলি মেনে চলুন\\.\n\n"
            "*আল্লাহ আমাদের সবাইকে উপকারী জ্ঞানে সমৃদ্ধ করুন!*"
        ),
        "about_us_title": "💡 *AlimAI এবং আমাদের লক্ষ্য*",
        "about_us_content": (
            "AlimAI হলো একটি উদ্ভাবনী টেলিগ্রাম বট, যা ইসলামী জ্ঞানের বিশাল সমুদ্রে বিচরণ করার জন্য আপনার বিশ্বস্ত ডিজিটাল সঙ্গী হিসাবে ডিজাইন করা হয়েছে\\. আমাদের লক্ষ্য হলো পবিত্র কোরআন এবং সহীহ সুন্নাহর ভিত্তিতে নির্ভুল, নির্ভরযোগ্য এবং আধ্যাত্মিকভাবে uplifting উত্তর প্রদান করা\\.\n\n"
            "*আমাদের লক্ষ্য:*\n"
            "আমরা এমন একটি বিশ্বের স্বপ্ন দেখি যেখানে খাঁটি ইসলামী জ্ঞান সকলের জন্য সহজলভ্য এবং তাৎক্ষণিক হবে\\. AlimAI এর লক্ষ্য হলো জ্ঞান অন্বেষণকারী এবং ইসলামের সমৃদ্ধ ঐতিহ্যের মধ্যে ব্যবধান দূর করা, অত্যাধুনিক AI প্রযুক্তির মাধ্যমে আমাদের দ্বীনের গভীর উপলব্ধি বৃদ্ধি করা, আর এ সবই কঠোরভাবে ইসলামী নীতি ও আলেমদের আদব মেনে চলবে\\.\n\n"
            "আমরা বিশ্বাস করি যে, প্রযুক্তি, যখন বিশুদ্ধ উদ্দেশ্য দ্বারা পরিচালিত হয়, তখন তা কল্যাণ ও জ্ঞান ছড়িয়ে দেওয়ার জন্য একটি শক্তিশালী হাতিয়ার হতে পারে\\. আমরা ক্রমাগত উন্নতির জন্য প্রতিশ্রুতিবদ্ধ, যাতে AlimAI বিশ্বজুড়ে মুসলমানদের জন্য একটি উপকারী এবং নির্ভরযোগ্য সম্পদ হিসাবে টিকে থাকে\\.\n\n"
            "আল্লাহ আমাদের এই ক্ষুদ্র প্রচেষ্টা কবুল করুন এবং এই প্রকল্পটিকে উম্মাহর জন্য অশেষ বরকতের উৎস বানান\\."
        )
    },
    "ar": {
        "start_welcome": "السلام عليكم ورحمة الله، *{user_name}*!\n\nأنا عليم الذكاء الاصطناعي، رفيقك الرقمي لتحصيل المعرفة الإسلامية. أنا مستعد لمساعدتك في مختلف المواضيع الإسلامية بما في ذلك *القرآن والحديث والفقه والمنطق والتاريخ والعلوم*.\n\nأرسل لي أسئلتك. يمكنك إعادة تشغيل أو إعادة تعيين المحادثة باستخدام /reset أو /restart.",
        "start_buttons_prompt": "استكشف ميزات عليم الذكاء الاصطناعي:",
        "user_guideline_button": "دليل المستخدم 📖",
        "about_us_button": "عنا 💡",
        "reset_success": "♻️ تم إعادة تعيين المحادثة. يمكنك الآن طرح سؤال جديد، إن شاء الله.",
        "language_prompt": "🌐 الرجاء اختيار لغتك المفضلة:",
        "only_islamic": "أخي/أختي الكريمة، يمكنني فقط مساعدتك في الأمور المتعلقة بالمعرفة الإسلامية. إن شاء الله، يمكنك طرح سؤال إسلامي آخر.",
        "lang_set_to": "تم تعيين اللغة إلى *{lang_name}*.",
        "model_prompt": "🧠 الرجاء اختيار نموذج الذكاء الاصطناعي المفضل لديك لتوليد النصوص:",
        "model_set_to": "تم تعيين نموذج الذكاء الاصطناعي إلى *{model_name}*.",
        "image_model_prompt": "🖼️ الرجاء اختيار نموذج لتوليد الصور:",
        "image_generation_start": "يتم توليد صورتك... يرجى الانتظار لحظة، إن شاء الله.",
        "image_generation_success": "هذه صورتك، إن شاء الله:",
        "image_generation_failed": "عذراً، لم أتمكن من توليد الصورة. يرجى المحاولة مرة أخرى أو بوصف مختلف.",
        "img_command_usage": "الرجاء استخدام الأمر هكذا: `/img [وصف صورتك]`",
        "no_image_description": "الرجاء تقديم وصف للصورة التي تريد توليدها. مثال: `/img قطة بخلفية مسجد`",
        "ask_command_usage": "الرجاء استخدام `/ask` متبوعًا بسؤالك. مثال: `/ask ما هي أركان الإسلام؟`",
        "api_error": "عذراً، أواجه مشكلة في الاتصال بخدمة الذكاء الاصطناعي الآن. يرجى المحاولة مرة أخرى لاحقاً، إن شاء الله.",
        "unknown_action": "إجراء غير معروف.",
        "guideline_title": "📖 *دليل مستخدم عليم الذكاء الاصطناعي*",
        "guideline_content": (
            "*مرحباً بك في عليم الذكاء الاصطناعي، رفيقك في المعرفة الإسلامية! إليك كيفية استخدامي:*\n\n"
            "1\\. *طرح الأسئلة:*\n"
            "   • *الدردشة الخاصة: * ما عليك سوى كتابة سؤالك مباشرةً\\.\n"
            "   • *دردشة المجموعة: * استخدم `/ask` متبوعًا بسؤالك \\(على سبيل المثال، `/ask ما هي أركان الإسلام؟`\\) أو الرد على إحدى رسائلي بسؤالك\\.\n\n"
            "2\\. *المواضيع المدعومة:*\n"
            "   أجيب بدقة على الأسئلة المتعلقة بـ *القرآن، الحديث، الفقه، السيرة، العقيدة، المنطق، التاريخ الإسلامي، والعلوم الإسلامية*\\.\n"
            "   *سأرفض بأدب الأسئلة خارج هذه المواضيع\\.*\n\n"
            "3\\. *اختيار اللغة:*\n"
            "   استخدم الأمر `/language` لاختيار لغتك المفضلة: الإنجليزية، البنغالية، أو العربية\\.\n\n"
            "4\\. *اختيار نموذج الذكاء الاصطناعي:*\n"
            "   استخدم الأمر `/model` لاختيار نموذج الذكاء الاصطناعي المفضل لديك لتوليد النصوص \\(مثل، جيميني، ديب سيك\\)\\.\n\n"
            "5\\. *توليد الصور:*\n"
            "   • استخدم الأمر `/image` لعرض نماذج توليد الصور المتاحة\\.\n"
            "   • لتوليد صورة، استخدم `/img` متبوعًا بالوصف \\(على سبيل المثال، `/img مسجد في الجنة`\\)\\.\n\n"
            "6\\. *إعادة تعيين المحادثة:*\n"
            "   استخدم `/reset` أو `/restart` لمسح سياق المحادثة الحالي والبدء من جديد\\.\n\n"
            "7\\. *التوجيه الإسلامي:*\n"
            "   سأسعى دائماً للإجابة بما يتوافق مع التعاليم والآداب الإسلامية\\. للأسئلة غير المناسبة، سأقدم توجيهاً إسلامياً بناءً على القرآن والحديث\\. يرجى احترام هذه الإرشادات\\.\n\n"
            "*نسأل الله أن يزيدنا علماً نافعاً!*"
        )
    }
}