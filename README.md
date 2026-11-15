# MLX RAG Studio

A local, on-device RAG workspace for YouTube videos, built with Streamlit, LangChain, FAISS and Apple MLX.

- Transcripts via `youtube-transcript-api` (any caption language the video really has)
- Chunks 1000/200, Qwen3-Embedding-0.6B (normalized) + FAISS, MMR retrieval (k=3, fetch_k=15, lambda=0.5)
- Qwen3-8B 4-bit through MLX-LM, streamed answers
- Video language and answer language are chosen independently

## Run locally (Apple Silicon Mac)

```bash
pip install -r requirements.txt
streamlit run rag1.py
```

The model folders are expected at the paths set at the top of `rag1.py`
(`EMBEDDING_MODEL_PATH`, `LLM_MODEL_PATH`). Models are not part of this repo.

## Streamlit Community Cloud

This app **will not run on Streamlit Community Cloud as is**: it needs Apple Silicon (MLX GPU inference)
and the two local model folders, and Community Cloud runs Linux with no access to them.
Deploying it there would need a different, hosted LLM and embedding backend.
