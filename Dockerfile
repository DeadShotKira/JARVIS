FROM python:3.11-slim

WORKDIR /app

COPY jarvis/requirements/requirements.txt /app/jarvis/requirements/requirements.txt
RUN pip install --no-cache-dir -r /app/jarvis/requirements/requirements.txt

COPY . /app

ENV JARVIS_CONFIG_PATH=/app/config.yaml

CMD ["python", "main.py"]
