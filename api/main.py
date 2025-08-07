# api/main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.simulator import run_one_episode, get_log

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev mode. Restrict in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Agentic Workflow RL API"}

@app.post("/simulate")
def simulate():
    return run_one_episode()

@app.get("/log")
def log():
    return get_log()
