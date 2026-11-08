FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY ml ./ml
COPY backend ./backend
RUN pip install --no-cache-dir .
RUN useradd --create-home research && mkdir /app/.data && chown research /app/.data
USER research
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
