from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import asyncio
import psutil
import random

from database import SessionLocal, engine, get_db
import models
import person1_predictor as predictor_advanced
import person3_algorithms as algorithms
import person2_rescheduler as rescheduler
import person4_simulator as simulator

models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Adaptive ML Workload Orchestrator — Phase 2")

scheduler_lock = asyncio.Lock()
CURRENT_ALGORITHM = "Priority"
LAST_RR_TASK_ID = None

async def scheduler_loop():
    global LAST_RR_TASK_ID
    while True:
        async with scheduler_lock:
            db = SessionLocal()
            try:
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent

                if rescheduler.detect_pressure(cpu_percent, ram_percent):
                    rescheduler.pause_lowest_priority_running(db)
                else:
                    rescheduler.resume_paused_tasks(db)

                pending = db.query(models.Task).filter(models.Task.status == "pending").all()
                ordered = []
                if pending:
                    if CURRENT_ALGORITHM == "SJF":
                        ordered = algorithms.order_sjf(pending, db)
                    elif CURRENT_ALGORITHM == "RoundRobin":
                        ordered = algorithms.order_round_robin(pending, LAST_RR_TASK_ID)
                    elif CURRENT_ALGORITHM == "FCFS":
                        ordered = algorithms.order_fcfs(pending)
                    else:
                        ordered = algorithms.order_priority(pending)

                next_task = ordered[0] if ordered else None
                if next_task and cpu_percent < 80 and ram_percent < 80:
                    cycle = simulator.detect_deadlock(db)
                    if cycle and next_task.task_id in cycle:
                        db.add(models.SchedulingDecision(
                            task_id=next_task.task_id, decision="rejected",
                            reason=f"deadlock_risk:{cycle}"))
                        db.commit()
                    else:
                        next_task.status = "running"
                        next_task.start_time = datetime.utcnow()
                        next_task.algorithm_used = CURRENT_ALGORITHM
                        LAST_RR_TASK_ID = next_task.task_id
                        db.add(models.SchedulingDecision(
                            task_id=next_task.task_id, decision="admitted", reason=CURRENT_ALGORITHM))
                        db.commit()
                        asyncio.create_task(simulate_task_execution(next_task.task_id))
            except Exception as e:
                db.rollback()
            finally:
                db.close()
        await asyncio.sleep(2)

async def simulate_task_execution(task_id: int):
    await asyncio.sleep(5)
    db = SessionLocal()
    try:
        task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
        if not task:
            return
        task.status = "completed"
        task.end_time = datetime.utcnow()

        req = db.query(models.ResourceRequirement).filter(models.ResourceRequirement.task_id == task_id).first()
        
        base_cpu = (req.predicted_cpu if req and req.predicted_cpu else 15.0) * random.uniform(0.9, 1.1)
        base_ram = (req.predicted_ram_mb if req and req.predicted_ram_mb else 200.0) * random.uniform(0.9, 1.1)

        if random.random() < 0.10:
            actual_cpu = base_cpu * 1.8
            actual_ram = base_ram * 1.8
        else:
            actual_cpu = base_cpu
            actual_ram = base_ram

        wait_time = (task.start_time - task.submit_time).total_seconds() if task.start_time else 0
        duration = (task.end_time - task.start_time).total_seconds() if task.start_time and task.end_time else 0

        is_anomaly = rescheduler.check_anomaly(db, task.task_type, actual_cpu, actual_ram)

        db.add(models.ExecutionHistory(
            task_id=task.task_id, actual_cpu=actual_cpu, actual_ram_mb=actual_ram,
            actual_duration_sec=duration, wait_time_sec=wait_time,
            algorithm_used=task.algorithm_used, is_anomaly=is_anomaly,
        ))
        db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler_loop())

@app.post("/submit_task/")
def submit_task(task_name: str, task_type: str, dataset_id: int, priority: int = 1, db: Session = Depends(get_db)):
    try:
        new_task = models.Task(task_name=task_name, task_type=task_type, dataset_id=dataset_id, priority=priority)
        db.add(new_task)
        db.flush()

        dataset = db.query(models.Dataset).filter(models.Dataset.dataset_id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        baseline = algorithms.predict_baseline(task_type, dataset.size_mb)
        advanced = predictor_advanced.predict_advanced(db, task_type, dataset.size_mb, baseline)

        req = models.ResourceRequirement(
            task_id=new_task.task_id,
            baseline_cpu=baseline["cpu"], baseline_ram_mb=baseline["ram"], baseline_duration_sec=baseline["time"],
            predicted_cpu=advanced["cpu"], predicted_ram_mb=advanced["ram"], predicted_duration_sec=advanced["time"],
        )
        db.add(req)
        db.commit()
        return {"message": "Task submitted", "task_id": new_task.task_id, "prediction_source": advanced["source"]}
    except Exception as e:
        db.rollback()
        raise e

@app.post("/strategy/")
def set_strategy(algorithm: str):
    global CURRENT_ALGORITHM
    if algorithm not in algorithms.ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm. Choose from {list(algorithms.ALGORITHMS)}")
    CURRENT_ALGORITHM = algorithm
    return {"message": f"Scheduling algorithm set to {algorithm}"}

@app.get("/analytics/")
def get_analytics(db: Session = Depends(get_db)):
    query = text("""
        SELECT t.task_type, AVG(e.wait_time_sec) as avg_wait,
               SUM(CASE WHEN e.is_anomaly THEN 1 ELSE 0 END) as anomalies
        FROM execution_history e
        JOIN tasks t ON e.task_id = t.task_id
        GROUP BY t.task_type
    """)
    result = db.execute(query).fetchall()
    return [{"task_type": r[0], "avg_wait": r[1], "anomalies": r[2]} for r in result]

@app.get("/strategy_comparison/")
def strategy_comparison(objective: str = "balanced", db: Session = Depends(get_db)):
    results = algorithms.compare_strategies(db)
    recommended = algorithms.recommend_strategy(results, objective)
    return {"all_strategies": results, "recommended": recommended}

@app.get("/whatif/ram/")
def whatif_ram(new_ceiling_percent: float, db: Session = Depends(get_db)):
    return simulator.whatif_reduce_ram_ceiling(db, new_ceiling_percent)

@app.get("/whatif/surge/")
def whatif_surge(extra_tasks: int, db: Session = Depends(get_db)):
    return simulator.whatif_task_surge(db, extra_tasks)

@app.get("/deadlock_check/")
def deadlock_check(db: Session = Depends(get_db)):
    cycle = simulator.detect_deadlock(db)
    return {"deadlock_detected": cycle is not None, "cycle": cycle}

@app.get("/scheduling_decisions/")
def scheduling_decisions(db: Session = Depends(get_db)):
    rows = (
        db.query(models.SchedulingDecision)
        .order_by(models.SchedulingDecision.timestamp.desc())
        .limit(50)
        .all()
    )
    return [{"task_id": r.task_id, "decision": r.decision, "reason": r.reason, "timestamp": r.timestamp} for r in rows]

@app.post("/setup/")
def setup_default(db: Session = Depends(get_db)):
    d = models.Dataset(name="ImageNet Sample", size_mb=500.0)
    db.add(d)
    db.commit()
    return {"message": "Setup complete", "dataset_id": d.dataset_id}
