FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

ADD . app/

RUN python app/app.py

EXPOSE 5000

CMD ["python", "app/server.py", "serve"]