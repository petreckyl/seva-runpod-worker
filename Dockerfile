FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
