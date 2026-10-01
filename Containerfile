FROM docker.io/library/python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY ["CMAPSS-remaining life.py", "/app/"]
COPY data/ /app/data/

WORKDIR /app/data

CMD ["python", "/app/CMAPSS-remaining life.py"]