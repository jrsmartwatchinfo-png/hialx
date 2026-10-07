import json
import os
from threading import Thread

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
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


INDEX_HTML = r'''<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma চ্যাটবট</title>
<style>
  :root { --bg:#f5f6fa; --card:#fff; --text:#1f2430; --muted:#6b7280; --accent:#4f6ef7; --user:#4f6ef7; --bot:#eef0f6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12141a; --card:#1b1e27; --text:#e8eaf0; --muted:#8b93a7; --bot:#262a36; }
  }
  * { box-sizing:border-box; }
  html,body { height:100%; margin:0; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,"Noto Sans Bengali","Kalpurush",sans-serif; display:flex; justify-content:center; }
  .app { width:100%; max-width:820px; height:100%; display:flex; flex-direction:column; background:var(--card); }
  header { padding:14px 18px; border-bottom:1px solid var(--bot); display:flex; align-items:center; justify-content:space-between; }
  header h1 { font-size:18px; margin:0; }
  header small { color:var(--muted); }
  button { font:inherit; cursor:pointer; }
  #clear { background:none; border:1px solid var(--muted); color:var(--muted); border-radius:8px; padding:6px 12px; }
  #log { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:85%; padding:10px 14px; border-radius:14px; line-height:1.6; white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:var(--user); color:#fff; border-bottom-right-radius:4px; }
  .bot { align-self:flex-start; background:var(--bot); border-bottom-left-radius:4px; }
  .err { color:#e5484d; }
  form { display:flex; gap:10px; padding:14px 18px; border-top:1px solid var(--bot); }
  textarea { flex:1; resize:none; max-height:140px; padding:10px 12px; border-radius:12px; border:1px solid var(--muted); background:var(--card); color:var(--text); font:inherit; }
  #send { background:var(--accent); color:#fff; border:0; border-radius:12px; padding:0 20px; }
  #send:disabled { opacity:.5; cursor:not-allowed; }
  .hint { text-align:center; color:var(--muted); margin:auto; }
</style>
</head>
<body>
<div class="app">
  <header>
    <div><h1>Gemma চ্যাটবট</h1><small>Google Gemma দ্বারা চালিত</small></div>
    <button id="clear" type="button">নতুন চ্যাট</button>
  </header>
  <div id="log"><div class="hint" id="hint">কিছু লিখে শুরু করুন 👇</div></div>
  <form id="form">
    <textarea id="input" rows="1" placeholder="আপনার বার্তা লিখুন..." autofocus></textarea>
    <button id="send" type="submit">পাঠান</button>
  </form>
</div>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
let history = [];

function add(role, text) {
  const hint = document.getElementById('hint'); if (hint) hint.remove();
  const d = document.createElement('div');
  d.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
  return d;
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = input.scrollHeight + 'px'; });

document.getElementById('clear').onclick = () => {
  history = []; log.innerHTML = '<div class="hint" id="hint">কিছু লিখে শুরু করুন 👇</div>';
};

form.addEventListener('submit', async e => {
  e.preventDefault();
  const text = input.value.trim(); if (!text) return;
  input.value = ''; input.style.height = 'auto';
  add('user', text); history.push({ role: 'user', content: text });
  const bubble = add('bot', '...'); let answer = '';
  send.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history })
    });
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = '';
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n'); buf = parts.pop();
      for (const p of parts) {
        const data = p.replace(/^data: /, '');
        if (data === '[DONE]') continue;
        const j = JSON.parse(data);
        if (j.error) { bubble.classList.add('err'); bubble.textContent = 'ত্রুটি: ' + j.error; }
        else { answer += j.text; bubble.textContent = answer; log.scrollTop = log.scrollHeight; }
      }
    }
    if (answer) history.push({ role: 'assistant', content: answer });
    else history.pop();
  } catch (err) {
    bubble.classList.add('err'); bubble.textContent = 'সার্ভারে সংযোগ করা যায়নি।'; history.pop();
  }
  send.disabled = false; input.focus();
});
</script>
</body>
</html>
'''


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
