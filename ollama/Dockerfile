FROM ollama/ollama

ARG OLLAMA_EMBEDDING_MODEL

RUN ollama serve & sleep 5 && ollama pull $OLLAMA_EMBEDDING_MODEL