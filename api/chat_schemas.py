from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    symptoms: list[str] = Field(default_factory=list, max_length=30)
    session_id: str | None = Field(default=None, max_length=100)

class RetrainRequest(BaseModel):
    dataset_path: str = "data/Training.csv"
