def main():
    print("Hello from project!")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
