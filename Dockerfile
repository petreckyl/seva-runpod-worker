FROM runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
