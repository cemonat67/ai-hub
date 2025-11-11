import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

# (OpenAI Python SDK 1.x)
# requirements.txt içinde: fastapi uvicorn openai
from openai import OpenAI

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Render'da Environment → Variables kısmına OPENAI_API_KEY eklemelisin.
    # Burada boşsa sadece uyarı veriyoruz; /api/chat çağrısında hata dönecek.
    print("[WARN] OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="AI Hub Backend", version="0.1.0")

# CORS (gerekirse domainini ekleyebilirsin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Routes ----------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Basit sağlık kontrolü (Render health check ve debug için)."""
    return {"ok": True, "provider": "openai" if OPENAI_API_KEY else "unset"}


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Klasör kökünde index.html varsa onu döner; yoksa basit bir mesaj döner."""
    idx = Path(__file__).with_name("index.html")
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AI Hub Backend</h1><p>Index bulunamadı.</p>")


@app.post("/api/chat")
async def chat(body: Dict[str, Any]):
    """
    Basit Chat endpoint'i.

    Gönderim örnekleri:
    - {"input": "Merhaba!", "model": "gpt-4o-mini"}
    - {"messages": [{"role":"user","content":"Merhaba"}], "model":"gpt-4o-mini"}

    Varsayılan model: gpt-4o-mini
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing")

    model = body.get("model") or "gpt-4o-mini"

    # Esneklik: tek "input" string geldiyse messages'a çevir
    if "messages" in body and isinstance(body["messages"], list):
        messages: List[Dict[str, str]] = body["messages"]
    else:
        user_text = body.get("input")
        if not isinstance(user_text, str) or not user_text.strip():
            raise HTTPException(status_code=400, detail="Provide 'input' (string) or 'messages' (list).")
        messages = [{"role": "user", "content": user_text.strip()}]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(body.get("temperature", 0.7)),
        )
        text = resp.choices[0].message.content
        return {"model": model, "output": text}
    except Exception as e:
        # Prod için hata mesajını sadeleştiriyoruz
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")


# ---------- Entry point ----------
if __name__ == "__main__":
    # Render 'PORT' değişkenini set ediyor; local için 8000'e düşer.
    port = int(os.environ.get("PORT", 8000))
    # 0.0.0.0 bağlamak DIŞ ERİŞİM için şart (Render)
    uvicorn.run(app, host="0.0.0.0", port=port)