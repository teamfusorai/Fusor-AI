import getpass
import os
import uvicorn
from fastapi import FastAPI
import data_ingestion

app = FastAPI()
app.include_router(data_ingestion.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
