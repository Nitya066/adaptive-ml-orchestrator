from datetime import datetime
from sqlalchemy.orm import Session
import models
import person1_predictor as predictor

HIGH_PRIORITY = 7
SUBTYPES = ["binary", "multiclass", "multilabel"]
FRACTIONS = [0.3, 0.6, 1.0]
# Fixed priorities for the 9-task benchmark
BATCH_PRIORITIES = [9, 3, 5, 1, 8, 7, 6, 2, 4]


def predict_baseline(task_type: str, dataset_size_mb: float) -> dict:
    base_cpu = 10.0
    base_ram = 50.0
    if task_type == "Training":
        return {"cpu": base_cpu * 2, "ram": base_ram * dataset_size_mb / 100, "time": dataset_size_mb / 50 * 2}
    if task_type == "Image Processing":
        return {"cpu": base_cpu * 1.5, "ram": base_ram * dataset_size_mb / 150, "time": dataset_size_mb / 50 * 1.5}
    return {"cpu": base_cpu, "ram": base_ram * dataset_size_mb / 200, "time": dataset_size_mb / 50}


def order_fcfs(tasks):
    return sorted(tasks, key=lambda t: t.submit_time)


def order_sjf(tasks, db: Session):
    def predicted_duration(t):
        req = db.query(models.ResourceRequirement).filter(models.ResourceRequirement.task_id == t.task_id).first()
        return req.predicted_duration_sec if req and req.predicted_duration_sec else 9999.0
    return sorted(tasks, key=predicted_duration)


def order_priority(tasks):
    return sorted(tasks, key=lambda t: -t.priority)


def apply_aging(db: Session, threshold_sec: float = 30.0) -> list:
    """
    Standard OS Aging technique to prevent starvation (Silberschatz OS Concept):
    Gradually increments the priority of tasks waiting in 'pending' status
    for longer than threshold_sec (capped at maximum priority 10).
    Returns list of task_ids whose priority was boosted.
    """
    now = datetime.utcnow()
    aged_tasks = []
    pending = db.query(models.Task).filter(models.Task.status == "pending").all()
    for t in pending:
        if t.submit_time and (now - t.submit_time).total_seconds() >= threshold_sec:
            cur_priority = t.priority if t.priority is not None else 1
            if cur_priority < 10:
                t.priority = min(10, cur_priority + 1)
                aged_tasks.append(t.task_id)
    if aged_tasks:
        db.commit()
    return aged_tasks


def order_round_robin(tasks, last_served_id=None):
    if not tasks:
        return tasks
    ids = [t.task_id for t in tasks]
    if last_served_id in ids:
        idx = ids.index(last_served_id)
        return tasks[idx + 1:] + tasks[: idx + 1]
    return tasks


ALGORITHMS = {
    "FCFS": order_fcfs,
    "SJF": order_sjf,
    "Priority": order_priority,
    "RoundRobin": order_round_robin,
}


def _standard_batch(db: Session):
    """
    Standard 9-task batch (3 subtypes x 3 sizes).
    Arrival times are staggered (0.0, 0.4, 0.8, ...) to evaluate preemption.
    """
    batch = []
    i = 0
    for subtype in SUBTYPES:
        ds = (
            db.query(models.Dataset)
            .join(models.Task, models.Task.dataset_id == models.Dataset.dataset_id)
            .filter(models.Task.subtype == subtype)
            .first()
        )
        for fraction in FRACTIONS:
            runs = (
                db.query(models.ExecutionHistory.actual_duration_sec)
                .join(models.Task, models.ExecutionHistory.task_id == models.Task.task_id)
                .filter(models.Task.subtype == subtype, models.Task.data_fraction == fraction)
                .all()
            )
            durations = [r[0] for r in runs if r[0] is not None]
            if not ds or not durations:
                return None
            effective_size = ds.size_mb * fraction
            baseline = predict_baseline("Classification", effective_size)
            pred = predictor.predict_advanced(
                db, "Classification", effective_size, baseline,
                n_features=ds.n_features, n_classes=ds.n_classes)
            batch.append({
                "id": i + 1,
                "actual": max(0.1, sum(durations) / len(durations)),
                "predicted": max(0.1, pred["time"]),
                "priority": BATCH_PRIORITIES[i],
                "arrival": i * 0.4,   # staggered arrival
            })
            i += 1
    return batch


def _simulate_non_preemptive(jobs, sort_key):
    """Simulates non-preemptive queue where available jobs at time t are sorted by sort_key."""
    current_time = 0.0
    pending = list(jobs)
    completed_waits = []

    while pending:
        available = [j for j in pending if j["arrival"] <= current_time]
        if not available:
            current_time = min(j["arrival"] for j in pending)
            available = [j for j in pending if j["arrival"] <= current_time]

        chosen = min(available, key=sort_key)
        pending.remove(chosen)
        wait = current_time - chosen["arrival"]
        completed_waits.append((chosen, max(0.0, wait)))
        current_time += chosen["actual"]

    return _calculate_wait_stats(completed_waits)


def _simulate_srtf(jobs):
    """Simulates preemptive Shortest Remaining Time First (SRTF)."""
    current_time = 0.0
    remaining = {j["id"]: j["actual"] for j in jobs}
    job_map = {j["id"]: j for j in jobs}
    completed_time = {}
    time_step = 0.05
    max_limit = sum(j["actual"] for j in jobs) + max(j["arrival"] for j in jobs) + 10.0

    while len(completed_time) < len(jobs) and current_time < max_limit:
        available_ids = [
            jid for jid, rem in remaining.items()
            if rem > 1e-4 and job_map[jid]["arrival"] <= current_time
        ]
        if not available_ids:
            unstarted = [j["arrival"] for j in jobs if j["id"] not in completed_time]
            current_time = min(unstarted) if unstarted else current_time + time_step
            continue

        # pick job with shortest predicted/actual remaining time
        chosen_id = min(available_ids, key=lambda jid: (remaining[jid], job_map[jid]["predicted"]))
        remaining[chosen_id] -= time_step
        current_time += time_step

        if remaining[chosen_id] <= 1e-4:
            remaining[chosen_id] = 0.0
            completed_time[chosen_id] = current_time

    completed_waits = [
        (job_map[jid], max(0.0, completed_time[jid] - job_map[jid]["arrival"] - job_map[jid]["actual"]))
        for jid in completed_time
    ]
    return _calculate_wait_stats(completed_waits)


def _simulate_priority_p(jobs):
    """Simulates Preemptive Priority (Priority-P)."""
    current_time = 0.0
    remaining = {j["id"]: j["actual"] for j in jobs}
    job_map = {j["id"]: j for j in jobs}
    completed_time = {}
    time_step = 0.05
    max_limit = sum(j["actual"] for j in jobs) + max(j["arrival"] for j in jobs) + 10.0

    while len(completed_time) < len(jobs) and current_time < max_limit:
        available_ids = [
            jid for jid, rem in remaining.items()
            if rem > 1e-4 and job_map[jid]["arrival"] <= current_time
        ]
        if not available_ids:
            unstarted = [j["arrival"] for j in jobs if j["id"] not in completed_time]
            current_time = min(unstarted) if unstarted else current_time + time_step
            continue

        # Highest priority runs first; preempts if a higher priority arrives
        chosen_id = min(available_ids, key=lambda jid: -job_map[jid]["priority"])
        remaining[chosen_id] -= time_step
        current_time += time_step

        if remaining[chosen_id] <= 1e-4:
            remaining[chosen_id] = 0.0
            completed_time[chosen_id] = current_time

    completed_waits = [
        (job_map[jid], max(0.0, completed_time[jid] - job_map[jid]["arrival"] - job_map[jid]["actual"]))
        for jid in completed_time
    ]
    return _calculate_wait_stats(completed_waits)


def _simulate_round_robin_slices(jobs, quantum=1.0):
    """Simulates Round Robin sharing the CPU with a time quantum."""
    current_time = 0.0
    queue = []
    remaining = {j["id"]: j["actual"] for j in jobs}
    job_map = {j["id"]: j for j in jobs}
    completed_time = {}
    arrived = set()

    all_jobs_sorted = sorted(jobs, key=lambda j: j["arrival"])

    while len(completed_time) < len(jobs):
        for j in all_jobs_sorted:
            if j["id"] not in arrived and j["arrival"] <= current_time:
                queue.append(j["id"])
                arrived.add(j["id"])

        if not queue:
            unarrived = [j for j in all_jobs_sorted if j["id"] not in arrived]
            if unarrived:
                current_time = unarrived[0]["arrival"]
                continue
            break

        jid = queue.pop(0)
        slice_time = min(quantum, remaining[jid])
        current_time += slice_time
        remaining[jid] -= slice_time

        # Check arrivals during this slice
        for j in all_jobs_sorted:
            if j["id"] not in arrived and j["arrival"] <= current_time:
                queue.append(j["id"])
                arrived.add(j["id"])

        if remaining[jid] > 1e-4:
            queue.append(jid)
        else:
            remaining[jid] = 0.0
            completed_time[jid] = current_time

    completed_waits = [
        (job_map[jid], max(0.0, completed_time[jid] - job_map[jid]["arrival"] - job_map[jid]["actual"]))
        for jid in completed_time
    ]
    return _calculate_wait_stats(completed_waits)


def _calculate_wait_stats(completed_waits):
    all_waits = [w for _, w in completed_waits]
    high = [w for j, w in completed_waits if j["priority"] >= HIGH_PRIORITY]
    return {
        "avg_wait_sec": round(sum(all_waits) / len(all_waits), 2) if all_waits else 0.0,
        "high_priority_wait_sec": round(sum(high) / len(high), 2) if high else None,
        "max_wait_sec": round(max(all_waits), 2) if all_waits else 0.0,
    }


def compare_strategies(db: Session, quantum: float = 1.0):
    """
    Compares all 6 algorithms on the same 9-job benchmark.
    Supports adjustable Round Robin time quantum.
    Returns list of algorithm statistics.
    """
    batch = _standard_batch(db)
    if not batch:
        return []

    return [
        {"algorithm": "FCFS", **_simulate_non_preemptive(batch, lambda j: j["arrival"])},
        {"algorithm": "SJF", **_simulate_non_preemptive(batch, lambda j: j["predicted"])},
        {"algorithm": "SRTF", **_simulate_srtf(batch)},
        {"algorithm": "Priority", **_simulate_non_preemptive(batch, lambda j: -j["priority"])},
        {"algorithm": "Priority-P", **_simulate_priority_p(batch)},
        {"algorithm": "RoundRobin", **_simulate_round_robin_slices(batch, quantum=quantum)},
    ]


def recommend_strategy(results, objective: str, custom_weights: dict = None):
    """
    Recommends optimal algorithm based on objective:
    - 'fastest': minimizes avg_wait_sec (typically SRTF or SJF)
    - 'priority': minimizes high_priority_wait_sec (typically Priority-P)
    - 'balanced': weighted sum of normalized metrics using custom_weights
    """
    if not results:
        return None
    if objective == "fastest":
        return min(results, key=lambda r: r["avg_wait_sec"])
    if objective == "priority":
        valid = [r for r in results if r["high_priority_wait_sec"] is not None]
        return min(valid, key=lambda r: r["high_priority_wait_sec"]) if valid else None

    # Balanced with customizable weights
    weights = custom_weights or {"wait": 0.34, "priority": 0.33, "max_wait": 0.33}
    w_wait = float(weights.get("wait", 0.34))
    w_prio = float(weights.get("priority", 0.33))
    w_max = float(weights.get("max_wait", 0.33))

    def norm(key):
        vals = [r[key] for r in results if r[key] is not None]
        worst = max(vals) if vals and max(vals) > 0 else 1.0
        return lambda r: (r[key] if r[key] is not None else worst) / worst

    n_wait = norm("avg_wait_sec")
    n_prio = norm("high_priority_wait_sec")
    n_max = norm("max_wait_sec")

    def score(r):
        return w_wait * n_wait(r) + w_prio * n_prio(r) + w_max * n_max(r)

    return min(results, key=score)
