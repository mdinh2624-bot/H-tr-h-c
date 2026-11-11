
import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Rose Study AI API")

# Khi triển khai, đặt FRONTEND_ORIGIN thành URL website của bạn.
frontend = os.getenv("FRONTEND_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

API_KEY = os.getenv("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    message: str
    mode: str = "tutor"
    system_instruction: str = ""
    image: str | None = None
    use_search: bool = False

@app.get("/")
def home():
    return {
        "name": "Rose Study AI",
        "status": "online"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Chưa cấu hình GEMINI_API_KEY"
        )

    try:
        client = genai.Client(api_key=API_KEY)

        contents = [
            types.Part.from_text(text=request.message)
        ]

        # Phân tích ảnh bài tập
        if request.image:
            data = request.image

            if "," in data and data.startswith("data:"):
                header, data = data.split(",", 1)
                mime_type = header.split(";")[0][5:]
            else:
                mime_type = "image/jpeg"

            image_bytes = base64.b64decode(data)

            if len(image_bytes) > 5 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="Ảnh vượt quá 5MB"
                )

            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            )

        instruction = request.system_instruction or (
            "Bạn là Rose Study AI, trợ lý học tập "
            "thân thiện, chính xác, giải thích bằng tiếng Việt."
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                temperature=0.5,
                max_output_tokens=4096
            )
        )

        return {
            "reply": response.text or "AI chưa tạo được câu trả lời."
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Lỗi xử lý AI: " + str(error)[:300]
        )
