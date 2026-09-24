from pydantic import BaseModel, Field

class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)

class PatientInput(BaseModel):
    age: int = Field(ge=0, le=130)
    gender: str
    blood_type: str
    medical_condition: str = ""
    admission_type: str = ""
    medication: str = ""
    test_results: str = ""
    date_of_admission: str | None = None
    date_of_discharge: str | None = None
    billing_amount: float | None = Field(default=None, ge=0)

class RetrainRequest(BaseModel):
    dataset_path: str = "data/hospital_training.csv"
