# backend/Dockerfile

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./immowelt_price_guide/backend /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]