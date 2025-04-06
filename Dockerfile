FROM mcr.microsoft.com/devcontainers/python:3.11

# Install base packages
RUN apt-get update && apt-get -y install --no-install-recommends \
    build-essential aria2 zstd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG PIXI_VERSION=v0.44.0

RUN curl -L -o /usr/local/bin/pixi -fsSL --compressed "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    && chmod +x /usr/local/bin/pixi \
    && pixi info
    

# Environment Variables
USER vscode
ENV DAGSTER_HOME "/home/vscode"

# Working Directory
WORKDIR /workspaces/datadex
