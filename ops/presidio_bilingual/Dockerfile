FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    presidio-analyzer \
    spacy \
    stanza

RUN python - <<'PY'
import stanza

stanza.download("en")
stanza.download("zh")
PY

COPY app.py /app/app.py

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
