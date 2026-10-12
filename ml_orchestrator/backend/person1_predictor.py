from sqlalchemy.orm import Session
from sklearn.linear_model import LinearRegression
import numpy as np
import models

MIN_SAMPLES_FOR_MODEL = 5

def predict_advanced(db: Session, task_type: str, dataset_size_mb: float, baseline: dict) -> dict:
    """
    Returns {"cpu", "ram", "time", "source"}.
    Uses a Simple Linear Regression Machine Learning model to predict resource usage 
    based on the dataset size, learning from historical execution data.
    """
    # 1. Fetch historical data (Execution History + Dataset Size)
    records = (
        db.query(models.ExecutionHistory, models.Dataset.size_mb)
        .join(models.Task, models.ExecutionHistory.task_id == models.Task.task_id)
        .join(models.Dataset, models.Task.dataset_id == models.Dataset.dataset_id)
        .filter(models.Task.task_type == task_type)
        .all()
    )

    if len(records) < MIN_SAMPLES_FOR_MODEL:
        return {**baseline, "source": "baseline-fallback"}

    # 2. Prepare Data for ML Model
    # X = Feature (Dataset Size in MB)
    # y = Target (The resource we want to predict)
    X = np.array([[size] for _, size in records])
    
    y_cpu = np.array([h.actual_cpu or 0 for h, _ in records])
    y_ram = np.array([h.actual_ram_mb or 0 for h, _ in records])
    y_time = np.array([h.actual_duration_sec or 0 for h, _ in records])

    # 3. Train the Linear Regression Models (This is true Machine Learning!)
    model_cpu = LinearRegression().fit(X, y_cpu)
    model_ram = LinearRegression().fit(X, y_ram)
    model_time = LinearRegression().fit(X, y_time)

    # 4. Predict the resources for the NEW incoming dataset size
    X_new = np.array([[dataset_size_mb]])
    
    # max(0, val) ensures the model doesn't accidentally predict negative time/RAM
    pred_cpu = max(0.0, model_cpu.predict(X_new)[0])
    pred_ram = max(0.0, model_ram.predict(X_new)[0])
    pred_time = max(0.0, model_time.predict(X_new)[0])

    return {
        "cpu": round(pred_cpu, 2), 
        "ram": round(pred_ram, 2), 
        "time": round(pred_time, 2), 
        "source": "linear-regression"
    }
