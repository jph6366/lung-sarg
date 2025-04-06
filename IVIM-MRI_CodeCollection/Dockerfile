FROM python:3.11-slim

WORKDIR /app

# Install base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY ivim/ /app/ivim/

RUN pip install --no-cache-dir .[dev]

COPY . /app/

ENTRYPOINT ["pytest" ,"-s"]
