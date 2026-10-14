FROM python:3.13.3-slim
WORKDIR /app
COPY requirements.lock ./
RUN --mount=type=secret,id=pip_ca \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --no-cache-dir --require-hashes -r requirements.lock
COPY src ./src
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "ozon_bot.main"]
