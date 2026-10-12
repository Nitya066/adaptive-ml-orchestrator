from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    dataset_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    size_mb = Column(Float)

class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"))
    priority = Column(Integer, default=1)
    status = Column(String, default="pending", index=True)
    algorithm_used = Column(String, default="Priority")
    submit_time = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

class ResourceRequirement(Base):
    __tablename__ = "resource_requirements"
    req_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id", ondelete="CASCADE"))
    baseline_cpu = Column(Float)
    baseline_ram_mb = Column(Float)
    baseline_duration_sec = Column(Float)
    predicted_cpu = Column(Float)
    predicted_ram_mb = Column(Float)
    predicted_duration_sec = Column(Float)

class ExecutionHistory(Base):
    __tablename__ = "execution_history"
    history_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    actual_cpu = Column(Float)
    actual_ram_mb = Column(Float)
    actual_duration_sec = Column(Float)
    wait_time_sec = Column(Float)
    algorithm_used = Column(String)
    is_anomaly = Column(Boolean, default=False)
    execution_time = Column(DateTime, default=datetime.utcnow)

class SchedulingDecision(Base):
    __tablename__ = "scheduling_decisions"
    decision_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    decision = Column(String)
    reason = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ResourceLock(Base):
    __tablename__ = "resource_locks"
    lock_id = Column(Integer, primary_key=True, index=True)
    resource_name = Column(String, nullable=False)
    held_by_task_id = Column(Integer, ForeignKey("tasks.task_id"), nullable=True)
    requested_by_task_id = Column(Integer, ForeignKey("tasks.task_id"), nullable=True)
