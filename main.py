import asyncio
import json
import os
from contextlib import asynccontextmanager
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
    # Set CHAT_ID so publish.py (called via Bash by the agent) knows which chat owns the file
    os.environ["CHAT_ID"] = chat_id
    # Set WORKSPACE_ROOT so the skill always writes artifacts to the project root
    os.environ["WORKSPACE_ROOT"] = WORKSPACE_ROOT

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
WORKSPACE_ROOT = os.path.abspath(os.getcwd())


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


# ── GET /api/chats/:id/artifacts ──────────────────────────────────────────────

@app.get("/api/chats/{chat_id}/artifacts")
async def list_artifacts(chat_id: str):
    if not db.get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return db.list_published_files(chat_id)


# ── GET /api/files/download/{file_path} ──────────────────────────────────────
# Download any file path under the workspace as an attachment.

@app.get("/api/files/download/{file_path:path}")
async def download_file(file_path: str):
    requested_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, file_path))

    if os.path.commonpath([WORKSPACE_ROOT, requested_path]) != WORKSPACE_ROOT:
        raise HTTPException(status_code=404, detail="File not found")

    if not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        requested_path,
        media_type="application/octet-stream",
        filename=os.path.basename(requested_path),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
