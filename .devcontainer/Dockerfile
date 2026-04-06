# Supports GitHub Codespace with Python, Node.js, MongoDB, and Marp

FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04 

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.14
RUN apt-get update && apt-get install -y \
    python3.14 \
    python3.14-venv \
    python3.14-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.14 as default for `python` only (keep system `python3` intact)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.14 1

# Bootstrap and upgrade pip for Python 3.14
RUN python3.14 -m ensurepip --upgrade && \
    python3.14 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Node.js 20 LTS (required for modern Marp CLI)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Marp CLI globally
RUN npm install -g @marp-team/marp-cli
RUN npm install -g markdownlint-cli

# Install additional system utilities
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    jq \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

