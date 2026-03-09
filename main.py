import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)


# ── In-memory store ───────────────────────────────────────────────────────────

chats: dict[str, dict] = {}
messages: dict[str, list] = {}
sessions: dict[str, ClaudeSDKClient] = {}
locks: dict[str, asyncio.Lock] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model="claude-haiku-4-5",
        cwd=os.getcwd(),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep",
                       "WebSearch", "WebFetch", "AskUserQuestion", "Skill"],
        setting_sources=["project"],
        permission_mode="acceptEdits",
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Lifespan: clean up all sessions on shutdown ───────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for client in list(sessions.values()):
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass
    sessions.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)


# ── GET / (health check for readiness probe) ──────────────────────────────────

@app.get("/")
async def health():
    return {"status": "ok"}


# ── GET /api/chats ────────────────────────────────────────────────────────────

@app.get("/api/chats")
async def list_chats():
    return list(chats.values())


# ── POST /api/chats ───────────────────────────────────────────────────────────

class CreateChatBody(BaseModel):
    title: str | None = None


@app.post("/api/chats", status_code=201)
async def create_chat(body: CreateChatBody = CreateChatBody()):
    chat_id = str(uuid.uuid4())
    chats[chat_id] = {"id": chat_id, "title": body.title or "New Chat", "created_at": now()}
    messages[chat_id] = []
    locks[chat_id] = asyncio.Lock()

    # Start a persistent ClaudeSDKClient — each query() call on the same
    # client automatically continues the session, no manual ID tracking needed.
    client = ClaudeSDKClient(options=make_options())
    await client.__aenter__()
    sessions[chat_id] = client

    return chats[chat_id]


# ── GET /api/chats/:id ────────────────────────────────────────────────────────

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chats[chat_id]


# ── DELETE /api/chats/:id ─────────────────────────────────────────────────────

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found")
    client = sessions.pop(chat_id, None)
    if client:
        await client.__aexit__(None, None, None)
    chats.pop(chat_id)
    messages.pop(chat_id)
    locks.pop(chat_id, None)
    return {"success": True}


# ── GET /api/chats/:id/messages ───────────────────────────────────────────────

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found")
    return messages[chat_id]


# ── POST /api/chats/:id/messages  (SSE) ──────────────────────────────────────

class SendMessageBody(BaseModel):
    content: str


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, body: SendMessageBody):
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found")
    client = sessions.get(chat_id)
    if not client:
        raise HTTPException(status_code=404, detail="Session not found")

    messages[chat_id].append({
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": body.content,
        "created_at": now(),
    })

    async def stream():
        assistant_chunks: list[str] = []
        # Lock prevents a second message being sent while one is in flight.
        async with locks[chat_id]:
            try:
                await client.query(body.content)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                assistant_chunks.append(block.text)
                                yield sse({"type": "text", "content": block.text})
                            elif hasattr(block, "name"):
                                yield sse({"type": "tool", "name": block.name})
                    elif isinstance(msg, ResultMessage):
                        cost = (
                            f"${msg.total_cost_usd:.4f}"
                            if msg.total_cost_usd is not None
                            else "N/A"
                        )
                        yield sse({"type": "done", "cost": cost})
            except Exception as e:
                yield sse({"type": "error", "error": str(e)})
            finally:
                if assistant_chunks:
                    messages[chat_id].append({
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": "".join(assistant_chunks),
                        "created_at": now(),
                    })

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
