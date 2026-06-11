# ──────────────────────────────────────────────────────────
# Text Data Compliance Checker – Dockerfile
# ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps (TruffleHog, ScanCode, spaCy model)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install TruffleHog v3 (Go binary)
RUN curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b /usr/local/bin

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . /app/text/

# Create output directory
RUN mkdir -p /app/compliance_output

# Environment
ENV PYTHONPATH=/app
ENV COMPLIANCE_WORK_DIR=/app/compliance_output
ENV COMPLIANCE_OPA_URL=http://opa:8181
ENV COMPLIANCE_QWEN_GUARD_ENABLED=false
ENV COMPLIANCE_OPA_ENABLED=true

EXPOSE 9000

CMD ["uvicorn", "text.server:app", "--host", "0.0.0.0", "--port", "9000"]
