import json
import os
from threading import Thread

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "api").lower()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

app = FastAPI(title="Gemma Chatbot")


class Message(BaseModel):
    role: str  # "user" অথবা "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


# ---------------------------------------------------------------- API মোড
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        key = os.getenv("GOOGLE_API_KEY")
        if not key or "key" in key and "বসান" in key:
            raise RuntimeError(".env ফাইলে GOOGLE_API_KEY দিন")
        _client = genai.Client(api_key=key)
    return _client


def stream_api(messages):
    from google.genai import types

    model = os.getenv("GEMMA_API_MODEL", "gemma-3-27b-it")
    contents = []
    for i, m in enumerate(messages):
        text = m.content
        # Gemma system instruction সমর্থন করে না, তাই প্রথম মেসেজের সাথে জুড়ে দিচ্ছি
        if i == 0 and m.role == "user":
            text = f"{SYSTEM_PROMPT}\n\n{text}"
        role = "model" if m.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    for chunk in _get_client().models.generate_content_stream(
        model=model, contents=contents
    ):
        if chunk.text:
            yield chunk.text


# -------------------------------------------------------------- লোকাল মোড
_local = {}


def _load_local():
    if _local:
        return _local
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = os.getenv("GEMMA_LOCAL_MODEL", "google/gemma-3-1b-it")
    token = os.getenv("HF_TOKEN") or None
    _local["tok"] = AutoTokenizer.from_pretrained(name, token=token)
    _local["model"] = AutoModelForCausalLM.from_pretrained(
        name, token=token, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return _local


def stream_local(messages):
    from transformers import TextIteratorStreamer

    s = _load_local()
    tok, model = s["tok"], s["model"]
    chat = [{"role": m.role, "content": m.content} for m in messages]
    chat[0]["content"] = f"{SYSTEM_PROMPT}\n\n{chat[0]['content']}"
    inputs = tok.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(
        target=model.generate,
        kwargs=dict(**inputs, streamer=streamer, max_new_tokens=1024, do_sample=True, temperature=0.7),
        daemon=True,
    ).start()
    for piece in streamer:
        yield piece


# ------------------------------------------------------------------ রুট
@app.post("/api/chat")
def chat(req: ChatRequest):
    gen = stream_api if PROVIDER == "api" else stream_local

    def event_stream():
        try:
            for piece in gen(req.messages):
                yield f"data: {json.dumps({'text': piece})}\n\n"
        except Exception as e:  # ব্যবহারকারীকে ত্রুটি দেখানো
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
