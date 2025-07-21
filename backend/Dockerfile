# ---- Base Stage ----
FROM python:3.11-slim-bookworm as base
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# ---- Builder Stage ----
FROM base as builder

# This makes the CPU version the default if no argument is provided
ARG REQUIREMENTS_FILE=requirements-cpu.txt

WORKDIR /app

RUN pip install --upgrade pip

# CHANGE THIS: Remove 'backend/' prefix
COPY requirements-base.txt ./

# CHANGE THIS: Remove 'backend/' prefix from the variable path
COPY ${REQUIREMENTS_FILE} ./requirements.txt

# Now pip can find both files and install correctly
RUN pip install --no-cache-dir -r requirements.txt --target /app/deps

# CHANGE THIS: Copy the current directory (which is now 'backend')
COPY . .

RUN mkdir -p /app/data
RUN mkdir -p static
RUN touch .env


# ---- Final Stage ----
FROM gcr.io/distroless/python3-debian12 as final

WORKDIR /app
COPY --from=builder --chown=nonroot:nonroot /app /app

ENV PYTHONPATH=/app/deps
USER nonroot
EXPOSE 9090

ENTRYPOINT ["python3"]
CMD ["start.py"]