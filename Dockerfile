FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install OS tools + Node.js 20 (apt ships an old version; Claude Code requires 18+)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code@2.1.72

# Install Python dependencies (cached layer, runs as root)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source and skill files
COPY main.py db.py publish.py ./
COPY .claude/ ./.claude/

# Give ownership to non-root user 1000 before switching
RUN chown -R 1000:1000 /app

ENV PATH="/app/.venv/bin:$PATH"
# Set HOME so the Claude CLI can write session data under /app
ENV HOME=/app

USER 1000

EXPOSE 8888

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888", "--log-level", "info"]
