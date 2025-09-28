FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONPATH="/app:/opt/deps"

RUN pip install --upgrade pip
COPY . .
RUN pip install --no-cache-dir -r requirements.txt --target /opt/deps

RUN mkdir -p /app/data
RUN mkdir -p static
RUN touch .env


FROM gcr.io/distroless/python3-debian12 AS final

WORKDIR /app
COPY --from=builder --chown=nonroot:nonroot /app /app
COPY --from=builder --chown=nonroot:nonroot /opt/deps /app/deps

ENV PYTHONPATH=/app/deps
USER nonroot
EXPOSE 9090

ENTRYPOINT ["python3"]
CMD ["start.py"]