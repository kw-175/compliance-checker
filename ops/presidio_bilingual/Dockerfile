FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface
ENV PII_ENABLE_PRESIDIO=true
ENV PII_ENABLE_REGEX_RULES=true
ENV PII_ENABLE_GLINER=false
ENV PII_GLINER_MODEL=urchade/gliner_multi_pii-v1
ENV PII_GLINER_THRESHOLD=0.50

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    presidio-analyzer \
    spacy \
    stanza \
    gliner

RUN python - <<'PY'
import stanza

stanza.download("en")
stanza.download("zh")
PY

COPY app.py /app/app.py

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
