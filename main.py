from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root before any app module is imported
load_dotenv(Path(__file__).parent / ".env")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
