# Thumbnail API

A lightweight FastAPI service for generating and manipulating thumbnails.  
Designed with speed, flexibility, and modern web integrations in mind.

---

## Features

- 🚀 Built with [FastAPI](https://fastapi.tiangolo.com/) for high performance
- 🖼️ Supports image processing via [OpenCV](https://opencv.org/) and [Pillow](https://python-pillow.org/)
- ⚙️ Configurable via `.env`
- 🔄 Hot-reload during development with `uvicorn` + `watchfiles`
- 📦 Dependency management with `requirements.txt`

---

## Tech Stack

- **Framework:** FastAPI + Starlette
- **Image Processing:** OpenCV, Pillow, NumPy
- **Validation:** Pydantic
- **Server:** Uvicorn (ASGI)
- **Config:** python-dotenv, YAML

---

## Installation

```bash
git clone https://github.com/<your-username>/thumbnail-api.git
cd thumbnail-api
python -m venv .venv
source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

## Running
Running the below will expose the API on `http://localhost:8000`.

Visit `http://localhost:8000/docs` for swagger docs.

```bash
uvicorn app.main:app --reload
```
