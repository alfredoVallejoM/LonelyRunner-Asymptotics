import sys
import time
import csv
import os
import random
import numpy as np
import multiprocessing as mp
import gc
from functools import cmp_to_key
from fractions import Fraction


# =====================================================================
# 1. GENERATORS (Number Theory Sequences)
# =====================================================================
def get_fibonacci(n):
    """Generates the first n Fibonacci numbers starting from 2, 3."""
    fibs = [2, 3]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def get_first_n_primes(n):
    """Generates the first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


# =====================================================================
# 2. EXACT LAZY ENGINE (For Human-Scale Regimes)
# =====================================================================
def compare_events(e1, e2):
    """Cross-multiplication to sort fractions without floating-point errors."""
    cross1 = e1[0] * e2[1]
    cross2 = e2[0] * e1[1]
    if cross1 < cross2:
        return -1
    elif cross1 > cross2:
        return 1
    else:
        if e1[2] > e2[2]:
            return -1
        elif e1[2] < e2[2]:
            return 1
        return 0


def calculate_exact_lebesgue(k, velocities):
    """Calculates the exact Lebesgue measure using an integer-scaled sweep-line."""
    events = []
    for v in velocities:
        den = k * v
        for m in range(v):
            num_start = m * k - 1
            num_end = m * k + 1
            if num_start < 0:
                events.append((num_start + den, den, 1))
                events.append((den, den, -1))
                events.append((0, den, 1))
                events.append((num_end, den, -1))
            elif num_end > den:
                events.append((num_start, den, 1))
                events.append((den, den, -1))
                events.append((0, den, 1))
                events.append((num_end - den, den, -1))
            else:
                events.append((num_start, den, 1))
                events.append((num_end, den, -1))

    events.sort(key=cmp_to_key(compare_events))
    lonely_measure = Fraction(0)
    lonely_points_count = 0
    current_coverage = 0
    last_num, last_den = 0, 1

    for num, den, event_type in events:
        if current_coverage == 0:
            current_pos = Fraction(num, den)
            last_pos = Fraction(last_num, last_den)
            if current_pos > last_pos:
                lonely_measure += current_pos - last_pos
            lonely_points_count += 1
        current_coverage += event_type
        last_num, last_den = num, den

    if current_coverage == 0:
        current_pos = Fraction(1, 1)
        last_pos = Fraction(last_num, last_den)
        if current_pos > last_pos:
            lonely_measure += current_pos - last_pos

    area = float(lonely_measure) * 100
    return (
        lonely_measure.numerator,
        lonely_measure.denominator,
        area,
        lonely_points_count,
    )


# =====================================================================
# 3. ERGODIC BIG-INT ENGINE (Immune to Float64 Blindness)
# =====================================================================
def calculate_ergodic_lebesgue(k, velocities, num_samples=250_000):
    """Monte Carlo integration using massive integers for hyper-fast growth regimes."""
    RESOLUTION = 10**150
    threshold = RESOLUTION // k

    raw_points = [random.randrange(RESOLUTION) for _ in range(num_samples)]
    points = np.array(raw_points, dtype=object)

    is_lonely = np.ones(num_samples, dtype=bool)

    for v in velocities:
        pos = (points * v) % RESOLUTION
        dist_to_int = np.minimum(pos, RESOLUTION - pos)
        hit_mask = dist_to_int < threshold
        is_lonely &= ~hit_mask.astype(bool)

        if not np.any(is_lonely):
            break

    area_pct = (np.sum(is_lonely) / num_samples) * 100

    # Aggressive memory cleanup for massive arrays
    del raw_points, points, is_lonely, pos, dist_to_int, hit_mask
    return -1, -1, area_pct, -1


# =====================================================================
# 4. ROUTER ENGINE
# =====================================================================
def process_regime(k, name, velocities):
    """Routes the calculation to the appropriate engine based on memory constraints."""
    MAX_EVENTS_LIMIT = 2_500_000
    t0 = time.time()
    total_v = sum(velocities) if len(velocities) < 100 else float("inf")

    if total_v < MAX_EVENTS_LIMIT:
        n, d, area, pts = calculate_exact_lebesgue(k, velocities)
        mode = "[Exact]   "
    else:
        n, d, area, pts = calculate_ergodic_lebesgue(k, velocities)
        mode = "[Ergodic] "

    t = time.time() - t0
    return n, d, area, pts, t, mode


# =====================================================================
# 5. HPC WORKER (CPU CORES)
# =====================================================================
def worker_process(task_queue, result_queue):
    # Increase digit limit for NumPy's BigInt arithmetic
    sys.set_int_max_str_digits(2000000)
    random.seed()

    while True:
        task = task_queue.get()
        if task is None:  # Poison Pill (Shutdown signal)
            break

        k, name = task

        try:
            if name == "Critical":
                velocities = [i + (k**2) for i in range(1, k)]
            elif name == "Resonant":
                velocities = [(i * 12) + 1 for i in range(1, k)]
            elif name == "Lacunary":
                velocities = [2**i for i in range(1, k)]
            elif name == "Squares":
                velocities = [i**2 for i in range(1, k)]
            elif name == "Fibonacci":
                velocities = get_fibonacci(k - 1)
            elif name == "Consecutive":
                velocities = [i for i in range(1, k)]
            elif name == "Primes":
                velocities = get_first_n_primes(k - 1)
            else:
                velocities = []

            n, d, area, pts, t, mode = process_regime(k, name, velocities)
            result_queue.put((k, name, n, d, area, pts, t, mode))

        except Exception as e:
            result_queue.put((k, name, -1, -1, -1.0, -1, 0.0, "[ERROR]   "))

        finally:
            if "velocities" in locals():
                del velocities
            gc.collect()


# =====================================================================
# 6. DEDICATED I/O WRITER
# =====================================================================
def writer_process(result_queue, csv_filename, total_tasks, completed_offset):
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists or os.stat(csv_filename).st_size == 0:
            writer.writerow(
                [
                    "k",
                    "Regime",
                    "Exact_Measure_Num",
                    "Exact_Measure_Den",
                    "Area_Percentage",
                    "Singular_Points",
                    "Time_seconds",
                ]
            )

        completed = 0
        while completed < total_tasks:
            result = result_queue.get()
            k, name, n, d, area, pts, t, mode = result

            writer.writerow([k, name, n, d, area, pts, t])
            file.flush()

            completed += 1
            real_total = completed + completed_offset
            print(
                f"[{real_total}] k={k:<4} {mode} {name:<12}: Area={area:>7.4f}% | T={t:>5.2f}s"
            )


# =====================================================================
# 7. MAIN ORCHESTRATOR
# =====================================================================
def get_completed_tasks(filename):
    """Reads the CSV to enable checkpointing and seamless resuming."""
    completed = set()
    if not os.path.isfile(filename):
        return completed

    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2 and row[0].isdigit():
                    # Only consider it successful if Area != -1.0
                    if float(row[4]) != -1.0:
                        completed.add((int(row[0]), row[1]))
    except Exception as e:
        print(f"⚠️ Warning reading history: {e}")

    return completed


def run_harvest_hpc(max_k=2000, target_cores=12):
    csv_filename = "unified_lonely_runner_topology.csv"
    start_k = 2

    completed_tasks = get_completed_tasks(csv_filename)
    completed_offset = len(completed_tasks)

    print("=" * 85)
    print(f"HPC MULTIPROCESSING SWEEP ({target_cores} CORES) - 7 REGIMES UNIFIED")
    if completed_offset > 0:
        print(f"🚀 Resuming: Found {completed_offset} completed tasks in the CSV.")
    print("=" * 85)

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # The 7 Fundamental Regimes of the Lonely Runner Topology
    regimes = [
        "Consecutive",
        "Primes",
        "Critical",
        "Resonant",
        "Lacunary",
        "Squares",
        "Fibonacci",
    ]

    tasks_to_do = []
    for k in range(start_k, max_k + 1):
        for r in regimes:
            if (k, r) not in completed_tasks:
                tasks_to_do.append((k, r))

    total_tasks = len(tasks_to_do)

    if total_tasks == 0:
        print(f"✅ All tasks up to k={max_k} are already completed!")
        return

    # Fill the global queue
    for task in tasks_to_do:
        task_queue.put(task)

    # Add Poison Pills to safely shutdown workers
    for _ in range(target_cores):
        task_queue.put(None)

    writer = mp.Process(
        target=writer_process,
        args=(result_queue, csv_filename, total_tasks, completed_offset),
    )
    writer.start()

    workers = []
    for i in range(target_cores):
        w = mp.Process(target=worker_process, args=(task_queue, result_queue))
        w.start()
        workers.append(w)

    for w in workers:
        w.join()

    writer.join()
    print("\n✅ HPC SWEEP COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    mp.freeze_support()
    # Ejecuta el script utilizando 12 núcleos
    run_harvest_hpc(max_k=2000, target_cores=12)
