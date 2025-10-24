FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy your project files
COPY pyproject.toml .
COPY app.py .
COPY . .

# Install dependencies using uv
RUN pip install uv
RUN uv sync --frozen --no-dev

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
