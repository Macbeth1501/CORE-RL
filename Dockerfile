# 1. Use a standard, stable Python 3.12 image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install basic system tools needed for the environment
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the root of the container
# Ensure you have a 'requirements.txt' in your local CORE-RL folder!
COPY requirements.txt .

# 5. Install Python dependencies using pip
# We include the core libraries explicitly to ensure they are present
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi pydantic openai openenv-core

# 6. Copy all your project files into the container
COPY . .

# 7. Set the Python path so the server can find the 'core_rl' module
ENV PYTHONPATH="/app"

# 8. Expose the port Hugging Face expects
EXPOSE 7860

# 9. Health check to let Hugging Face know the server is healthy
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# 10. Start the FastAPI server
# This points to core_rl/server/app.py which we just verified works locally
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]