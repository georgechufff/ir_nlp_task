FROM python:latest

RUN apt update && apt install python3-pip -y

WORKDIR /app

COPY . .

RUN pip install -r ./requirements.txt

RUN python -m nltk.downloader punkt

RUN python -c "import nltk; nltk.download('wordnet')" \
    && python -c "import nltk; nltk.download('stopwords')"

EXPOSE 8080

