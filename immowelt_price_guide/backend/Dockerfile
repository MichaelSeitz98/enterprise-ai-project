# backend/Dockerfile

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./backend/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY ./backend /app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
