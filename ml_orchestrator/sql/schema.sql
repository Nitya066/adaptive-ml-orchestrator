-- =========================================================
-- Adaptive ML Workload Orchestrator — Phase 2 Schema
-- Reference schema matching backend/models.py (the app itself runs on
-- SQLite via SQLAlchemy; this file shows the same design in plain SQL).
-- =========================================================

-- 1. Schema Design / Normalization -------------------------
CREATE TABLE datasets (
    dataset_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    size_mb FLOAT CHECK (size_mb > 0)
);

CREATE TABLE tasks (
    task_id SERIAL PRIMARY KEY,
    task_name VARCHAR(255) NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    dataset_id INT REFERENCES datasets(dataset_id),
    priority INT DEFAULT 1 CHECK (priority >= 1 AND priority <= 10),
    status VARCHAR(50) DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'paused', 'completed', 'failed')),
    algorithm_used VARCHAR(50),
    submit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    start_time TIMESTAMP,
    end_time TIMESTAMP
);

CREATE TABLE resource_requirements (
    req_id SERIAL PRIMARY KEY,
    task_id INT REFERENCES tasks(task_id) ON DELETE CASCADE,
    -- Person 3's baseline heuristic
    baseline_cpu FLOAT CHECK (baseline_cpu >= 0 AND baseline_cpu <= 100),
    baseline_ram_mb FLOAT CHECK (baseline_ram_mb >= 0),
    baseline_duration_sec FLOAT CHECK (baseline_duration_sec >= 0),
    -- Person 1's advanced (regression-based) prediction
    predicted_cpu FLOAT CHECK (predicted_cpu >= 0 AND predicted_cpu <= 100),
    predicted_ram_mb FLOAT CHECK (predicted_ram_mb >= 0),
    predicted_duration_sec FLOAT CHECK (predicted_duration_sec >= 0)
);

CREATE TABLE execution_history (
    history_id SERIAL PRIMARY KEY,
    task_id INT REFERENCES tasks(task_id),
    actual_cpu FLOAT,
    actual_ram_mb FLOAT,
    actual_duration_sec FLOAT,
    wait_time_sec FLOAT,
    algorithm_used VARCHAR(50),
    is_anomaly BOOLEAN DEFAULT FALSE,       -- Person 2
    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Person 2 + Person 4: audit log of every admit/pause/resume/reject decision
CREATE TABLE scheduling_decisions (
    decision_id SERIAL PRIMARY KEY,
    task_id INT REFERENCES tasks(task_id),
    decision VARCHAR(50),                   -- admitted | paused | resumed | rejected
    reason VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Person 4: who holds / wants which resource (used for the deadlock check)
CREATE TABLE resource_locks (
    lock_id SERIAL PRIMARY KEY,
    resource_name VARCHAR(100) NOT NULL,
    held_by_task_id INT REFERENCES tasks(task_id),
    requested_by_task_id INT REFERENCES tasks(task_id)
);

-- 2. Constraints ------------------------------------------------
-- CHECK and NOT NULL constraints are included directly in the tables above
-- (e.g. priority must be 1-10, status must be one of a fixed set, RAM can't be negative).

-- 3. Indexing ------------------------------------------------
-- Speeds up the scheduler's constant polling for pending/running tasks.
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_type ON tasks(task_type);
CREATE INDEX idx_decisions_task ON scheduling_decisions(task_id);

-- 4. Views ----------------------------------------------------
-- A pre-joined view so the app doesn't have to repeat the same JOIN everywhere.
CREATE VIEW task_resource_view AS
SELECT
    t.task_id, t.task_name, t.task_type, t.status, t.algorithm_used,
    d.name AS dataset_name, d.size_mb,
    r.baseline_cpu, r.baseline_ram_mb,
    r.predicted_cpu, r.predicted_ram_mb
FROM tasks t
JOIN datasets d ON t.dataset_id = d.dataset_id
JOIN resource_requirements r ON t.task_id = r.task_id;
