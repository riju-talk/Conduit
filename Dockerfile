FROM python:3.12.10-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "mcp.main:app", "--host", "0.0.0.0", "--port", "8000"]