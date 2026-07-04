# metrics/reports.py

import statistics
import csv
from pathlib import Path

def _fitness_list(poblacion):
    return [ind.get("fitness", 0.0) for ind in poblacion]

def compute_stats(poblacion):
    fits = _fitness_list(poblacion)
    if not fits:
        return {"count":0,"mean":0.0,"std":0.0,"min":0.0,"p25":0.0,"median":0.0,"p75":0.0,"max":0.0}
    srt = sorted(fits)
    n = len(srt) - 1
    return {
        "count": len(fits),
        "mean": float(statistics.mean(fits)),
        "std": float(statistics.pstdev(fits)) if len(fits) > 1 else 0.0,
        "min": float(srt[0]),
        "median": float(statistics.median(srt)),
        "max": float(srt[-1]),
        "p25": float(srt[int(0.25*n)]),
        "p75": float(srt[int(0.75*n)]),
    }

def init_metrics_files(outdir: Path):
    csv_path = outdir / "metrics_gen.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "generation","count","mean","std","min","median","max","p25","p75","duration_sec"
            ])
    return csv_path

def append_metrics(outdir: Path, generation: int, poblacion, duration_sec: float):
    stats = compute_stats(poblacion)
    csv_path = init_metrics_files(outdir)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            generation, stats["count"],
            f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
            f"{stats['min']:.6f}", f"{stats['median']:.6f}", f"{stats['max']:.6f}",
            f"{stats['p25']:.6f}", f"{stats['p75']:.6f}",
            f"{duration_sec:.6f}"
        ])