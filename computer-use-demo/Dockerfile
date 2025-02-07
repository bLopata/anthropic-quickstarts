FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY computer_use_demo/requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the package
COPY computer_use_demo computer_use_demo/

# Create workspace and logs directories with proper permissions
RUN mkdir -p /app/logs \
    && chown -R 1000:1000 /app/computer_use_demo /app/logs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Set user to non-root
USER 1000

# Run the agent
CMD ["python", "-m", "computer_use_demo.loop"]
