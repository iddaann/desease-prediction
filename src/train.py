"""Train the HealthPredict symptom-to-disease model."""
from api.disease_ml import train

if __name__ == "__main__":
    result = train("data/Training.csv")
    print("Model tersimpan.")
    print(result)
