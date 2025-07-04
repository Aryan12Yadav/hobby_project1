FROM python:3.10.13

WORKDIR /app

COPY requirements.txt .


RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8080


CMD ["gunicorn", "--workers=1", "--bind=0.0.0.0:8080", "app:app"]
