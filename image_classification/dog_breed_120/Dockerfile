FROM python:3.9.2

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV HOME /app

CMD ["python3", "/app/app.py"]