# InvoiceIQ application container. Ollama runs as a SEPARATE service
# (see docker-compose.yml) - bundling a multi-GB local LLM runtime and
# its model weights into the app image would make every rebuild huge
# and slow, for no benefit (the model doesn't change when app code does).

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data data/invoices logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
