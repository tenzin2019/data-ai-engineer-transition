from fastapi import FastAPI
from pydantic import BaseModel


# Declaring FastAPI Obj
app = FastAPI()

# Health check route
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI + Docker!"}

# Example prediction input
class LoanInput(BaseModel):
    age: int
    income: float
    loan_amount: float

# Dummy model route for testing
@app.post("/predict")
def predict(input: LoanInput):
    # Dummy prediction logic
    score = (input.income - input.loan_amount) / input.age
    return {"default_risk_score": round(score, 2)}
