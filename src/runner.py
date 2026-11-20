"""Run a list of commands (one per line, e.g. the output of controls.py / scaling.py / sweep.py / refit.py `plan`) as concurrent single-GPU jobs instead of torchrun/DDP.
`torchrun --nproc_per_node=N x.py ...` lines are rewritten to plain `python x.py ...`; jobs are spread over --gpus with --per_gpu jobs each; finished training jobs (their pred_<year>.npz / manage_<year>.json exists)
are skipped, so the list can be re-run after an interruption; a new job starts only while enough RAM is free.
  python runner.py cmds.sh --jobs 4                     # 1 GPU, 4 concurrent processes
  python runner.py cmds.sh --gpus 0,1 --per_gpu 2 --min_free_gb 8 --logs logs/"""
import argparse, os, shlex, subprocess, sys, time
from pathlib import Path

def rewrite(line):
    parts = shlex.split(line)
    if parts[0] == "torchrun": parts = [sys.executable] + [p for p in parts[1:] if not p.startswith("--nproc_per_node")]
    elif parts[0] in ("python", "python3"): parts[0] = sys.executable
    return parts

def done_marker(parts):
    def val(k): return parts[parts.index(k) + 1] if k in parts and parts.index(k) + 1 < len(parts) else None
    script = next((Path(p).name for p in parts if p.endswith(".py")), ""); out, y = val("--out"), val("--test_year")
    if out and y and script == "train.py": return Path(out) / f"pred_{y}.npz"
    if out and y and script == "ppo_manage.py": return Path(out) / f"manage_{y}.json"
    return None

def free_gb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"): return int(ln.split()[1]) / 1e6
    except OSError: pass
    return None                                                                       # not Linux: no RAM guard

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmds"); ap.add_argument("--gpus", default="0"); ap.add_argument("--per_gpu", type=int, default=3); ap.add_argument("--min_free_gb", type=float, default=6.0); ap.add_argument("--logs", default="runner_logs"); ap.add_argument("--dry_run", action="store_true"); a = ap.parse_args()
    lines = [l.strip() for l in open(a.cmds) if l.strip() and not l.lstrip().startswith("#")]; gpus = a.gpus.split(","); jobs = []
    for l in lines:
        p = rewrite(l); m = done_marker(p)
        if m is not None and m.exists(): continue
        jobs.append((l, p))
    print(f"{len(lines)} commands, {len(lines) - len(jobs)} already done, {len(jobs)} to run on GPU(s) {gpus} x {a.per_gpu} jobs", flush=True)
    if a.dry_run:
        for l, p in jobs: print(" ".join(shlex.quote(x) for x in p))
        return 0
    Path(a.logs).mkdir(parents=True, exist_ok=True); ncpu = os.cpu_count() or 1; slots = len(gpus) * a.per_gpu; run, failed, n_done, t_all = [], [], 0, time.time(); queue = list(enumerate(jobs))
    while queue or run:
        for r in list(run):
            rc = r["proc"].poll()
            if rc is None: continue
            run.remove(r); r["log"].close(); n_done += 1
            if rc != 0: failed.append((r["line"], rc, r["logpath"])); print(f"FAILED rc={rc}: {r['line'][:120]}  (log {r['logpath']})", flush=True)
            else: print(f"done {n_done}/{len(jobs)} in {time.time() - r['t0']:.0f}s: {r['line'][:100]}", flush=True)
        while queue and len(run) < slots:
            fg = free_gb()
            if fg is not None and run and fg < a.min_free_gb: break                    # wait for RAM; always allow one job so the queue cannot stall
            gi = min(range(len(gpus)), key=lambda g: sum(1 for r in run if r["gpu"] == g)); i, (l, p) = queue.pop(0); logpath = Path(a.logs) / f"job_{i:04d}.log"; log = open(logpath, "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[gi], OMP_NUM_THREADS=str(max(1, ncpu // slots)), MKL_NUM_THREADS=str(max(1, ncpu // slots)), NUMBA_NUM_THREADS=str(max(1, ncpu // slots)))
            run.append(dict(proc=subprocess.Popen(p, stdout=log, stderr=subprocess.STDOUT, env=env), gpu=gi, line=l, t0=time.time(), log=log, logpath=str(logpath)))
        time.sleep(2)
    print(f"finished {n_done} jobs in {(time.time() - t_all) / 60:.1f} min, {len(failed)} failed", flush=True)
    if failed: Path(a.logs, "failed.txt").write_text("\n".join(f"{rc}\t{l}\t{lp}" for l, rc, lp in failed)); return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
