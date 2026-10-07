# Gemma চ্যাটবট (Ollama ছাড়া)

Python (FastAPI) + সাধারণ HTML চ্যাট UI + Google Gemma।

## চালানোর নিয়ম

1. Python 3.10+ ইনস্টল থাকতে হবে।
2. টার্মিনালে প্রজেক্ট ফোল্ডারে গিয়ে:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   pip install -r requirements.txt
   ```
3. `.env.example` ফাইলটির নাম বদলে `.env` করুন।
4. https://aistudio.google.com/apikey থেকে বিনামূল্যে API key নিয়ে `.env` এর `GOOGLE_API_KEY` এ বসান।
5. চালান:
   ```
   uvicorn app:app --reload
   ```
6. ব্রাউজারে খুলুন: http://127.0.0.1:8000

## দুটি মোড
- `PROVIDER=api` (ডিফল্ট): Google-এর সার্ভারে Gemma চলে, আপনার পিসি ভারী লাগে না।
- `PROVIDER=local`: Hugging Face থেকে Gemma নামিয়ে নিজের পিসিতে চলে। `requirements.txt` এর কমেন্ট করা তিনটি লাইন চালু করে ইনস্টল করুন, Hugging Face এ Gemma-র লাইসেন্স accept করে `HF_TOKEN` দিন। `gemma-3-1b-it` ছোট মডেল, সাধারণ পিসিতেও চলে।

## ফাইল
- `app.py` — ব্যাকএন্ড
- `static/index.html` — চ্যাট UI
