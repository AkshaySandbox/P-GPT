FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir langchain-huggingface

# Copy the application code
COPY --chown=user . .

# Create necessary directories for data persistence
RUN mkdir -p $HOME/app/data/Qdrantdb $HOME/app/data/pdf $HOME/app/data/metadata

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]