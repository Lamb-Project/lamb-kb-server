FROM python:3.11-slim-bookworm as base
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


FROM base as builder

ARG REQUIREMENTS_FILE=requirements-cpu.txt

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements-base.txt ./
COPY ${REQUIREMENTS_FILE} ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --target /app/deps

COPY . .

RUN mkdir -p /app/data
RUN mkdir -p static
RUN touch .env


FROM gcr.io/distroless/python3-debian12 as final

WORKDIR /app
COPY --from=builder --chown=nonroot:nonroot /app /app

ENV PYTHONPATH=/app/deps
USER nonroot
EXPOSE 9090

ENTRYPOINT ["python3"]
CMD ["start.py"]