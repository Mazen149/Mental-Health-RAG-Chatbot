## MLflow UI

Start the tracking UI with the project environment and the working local database copy:

```powershell
.\start_mlflow_ui.ps1
```

The launcher uses `.venv`, `mlflow.db.backup-before-repair`, and the local `D:\content\mlruns` compatibility junction for artifact access.

## FastAPI Chatbot

Run the FastAPI app from the repository root:

```powershell
pip install -e .
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Then POST to `/chat` with JSON:

```json
{
  "query": "i am depressed, what should i do?"
}
```

A simple health check endpoint is available at `/health`.
