FROM docker.io/library/python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends iproute2 iputils-ping && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY ["CMAPSS-remaining life.py", "/app/"]
COPY src/ /app/src/
COPY data/ /app/data/

WORKDIR /app

CMD ["python", "/app/CMAPSS-remaining life.py"]
