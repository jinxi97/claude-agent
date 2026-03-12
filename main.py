import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    PermissionResultAllow,
    ToolPermissionContext,
)

import db


# ── Artifacts directory ───────────────────────────────────────────────────────

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

logger = logging.getLogger("claude-agent")


# ── In-memory store (runtime only — not persisted) ───────────────────────────

sessions: dict[str, ClaudeSDKClient] = {}
locks: dict[str, asyncio.Lock] = {}
active_session_chat_id: str | None = None  # only 1 CLI subprocess at a time
pending_answers: dict[str, asyncio.Future] = {}  # chat_id → Future for AskUserQuestion


# ── Helpers ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a slide-making agent, which creates stunning, animation-rich HTML presentations that run entirely in the browser.
You are powered by the frontend-slides skill (https://github.com/zarazhangrui/frontend-slides) and Funky's agent workspace (https://funky.dev) \

When the user asks to create a presentation or slides, always invoke the frontend-slides skill."""


def make_can_use_tool(chat_id: str):
    """Return an async can_use_tool callback bound to a specific chat."""

    async def can_use_tool(
        tool_name: str,
        tool_input: dict,
        context: ToolPermissionContext,
    ) -> PermissionResultAllow:
        if tool_name == "AskUserQuestion":
            # Create a future that blocks until the user answers via POST /answer
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
            pending_answers[chat_id] = future

            try:
                answers = await future  # dict[str, str]  question → label
            finally:
                pending_answers.pop(chat_id, None)

            # Merge answers into the original tool input
            updated = {**tool_input, "answers": answers}
            return PermissionResultAllow(updated_input=updated)

        return PermissionResultAllow(updated_input=tool_input)

    return can_use_tool


def make_options(chat_id: str, resume: str | None = None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model="claude-haiku-4-5",
        cwd=os.getcwd(),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep",
                       "WebSearch", "WebFetch", "Skill"],
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        system_prompt=SYSTEM_PROMPT,
        resume=resume,
        can_use_tool=make_can_use_tool(chat_id),
    )


def get_lock(chat_id: str) -> asyncio.Lock:
    """Return (or lazily create) the per-chat asyncio lock."""
    if chat_id not in locks:
        locks[chat_id] = asyncio.Lock()
    return locks[chat_id]


async def ensure_session(chat_id: str) -> ClaudeSDKClient:
    """Return the session for chat_id, creating it lazily.
    Closes any other active session first — only 1 CLI subprocess at a time."""
    global active_session_chat_id

    if chat_id in sessions:
        return sessions[chat_id]

    # Close the previous session to free resources
    if active_session_chat_id and active_session_chat_id in sessions:
        old_client = sessions.pop(active_session_chat_id)
        try:
            await old_client.__aexit__(None, None, None)
        except Exception:
            pass
        logger.info("closed_session chat_id=%s (replaced by %s)",
                     active_session_chat_id, chat_id)

    # Resume previous session if this chat had one, otherwise start fresh
    resume_id = db.get_session_id(chat_id)
    client = ClaudeSDKClient(options=make_options(chat_id, resume=resume_id))
    await client.__aenter__()
    sessions[chat_id] = client
    active_session_chat_id = chat_id
    if resume_id:
        logger.info("resumed_session chat_id=%s session_id=%s", chat_id, resume_id)
    return client


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Lifespan: init DB on startup, clean up sessions on shutdown ──────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
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
    return db.list_chats()


# ── POST /api/chats ───────────────────────────────────────────────────────────

class CreateChatBody(BaseModel):
    title: str | None = None


@app.post("/api/chats", status_code=201)
async def create_chat(body: CreateChatBody = CreateChatBody()):
    return db.create_chat(body.title)


# ── GET /api/chats/:id ────────────────────────────────────────────────────────

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    chat = db.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


# ── DELETE /api/chats/:id ─────────────────────────────────────────────────────

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    global active_session_chat_id
    if not db.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    client = sessions.pop(chat_id, None)
    if client:
        await client.__aexit__(None, None, None)
    if active_session_chat_id == chat_id:
        active_session_chat_id = None
    locks.pop(chat_id, None)
    return {"success": True}


# ── GET /api/chats/:id/messages ───────────────────────────────────────────────

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    if not db.get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return db.get_messages(chat_id)


# ── POST /api/chats/:id/messages  (SSE) ──────────────────────────────────────

class SendMessageBody(BaseModel):
    content: str


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, body: SendMessageBody):
    if not db.get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

    client = await ensure_session(chat_id)

    # Persist user message
    db.add_message(chat_id, "user", body.content)

    async def stream():
        content_blocks: list[dict] = []
        # Lock prevents a second message being sent while one is in flight.
        async with get_lock(chat_id):
            try:
                await client.query(body.content)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                event = {"type": "text", "content": block.text}
                                content_blocks.append(event)
                                yield sse(event)
                            elif hasattr(block, "name"):
                                event = {"type": "tool", "name": block.name}
                                if hasattr(block, "input") and block.input:
                                    event["input"] = block.input
                                content_blocks.append(event)
                                yield sse(event)
                    elif isinstance(msg, ResultMessage):
                        db.set_session_id(chat_id, msg.session_id)
                        cost = (
                            f"${msg.total_cost_usd:.4f}"
                            if msg.total_cost_usd is not None
                            else "N/A"
                        )
                        yield sse({"type": "done", "cost": cost})
            except Exception as e:
                yield sse({"type": "error", "error": str(e)})
            finally:
                if content_blocks:
                    db.add_message(chat_id, "assistant", content_blocks)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── POST /api/chats/:id/answer  (unblock AskUserQuestion) ────────────────────

class AnswerBody(BaseModel):
    answers: dict[str, str]  # question text → selected option label


@app.post("/api/chats/{chat_id}/answer")
async def answer_question(chat_id: str, body: AnswerBody):
    future = pending_answers.get(chat_id)
    if not future:
        raise HTTPException(status_code=404, detail="No pending question for this chat")
    future.set_result(body.answers)
    return {"success": True}


# ── GET /api/artifacts/file/{filename} ───────────────────────────────────────
# Single endpoint to serve any artifact file — workspace ID in headers lets
# the router forward this to the correct Pod, just like every other API call.

@app.get("/api/artifacts/file/{filename:path}")
async def serve_artifact_file(filename: str):
    file_path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="text/html")


# ── GET /api/artifacts ────────────────────────────────────────────────────────
# Poll this at any phase — lists all HTML files recursively, including previews.

@app.get("/api/artifacts")
async def get_artifacts():
    files = []
    for dirpath, _, filenames in os.walk(ARTIFACTS_DIR):
        for filename in filenames:
            if filename.endswith(".html"):
                rel_path = os.path.relpath(os.path.join(dirpath, filename), ARTIFACTS_DIR)
                files.append({
                    "name": filename,
                    "path": rel_path,
                })
    return {"artifacts": files}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
