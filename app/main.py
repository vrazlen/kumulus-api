import os
from fastapi import FastAPI
from sqlalchemy import create_engine, text

# Initialize FastAPI app
app = FastAPI()

# Get the database URL from the environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

@app.on_event("startup")
def startup_event():
    """
    On startup, check the database connection.
    This confirms the environment variables are correctly set.
    """
    if DATABASE_URL is None:
        print("!!! DATABASE_URL environment variable not set. !!!")
        return

    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            # Execute a simple query to test the connection
            result = connection.execute(text("SELECT 1"))
            for row in result:
                print("--- Database connection successful. ---")
    except Exception as e:
        print(f"--- Database connection failed: {e} ---")


@app.get("/health")
def read_root():
    """
    Health check endpoint. If this is reachable, the app is live.
    """
    return {"status": "ok"}