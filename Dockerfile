FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN apt-get update -y && \
    apt-get install awscli -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]