"""Runner: torchrun rewrite, skipping finished jobs, concurrency over GPU slots with CUDA_VISIBLE_DEVICES set, failure reporting."""
import sys, subprocess, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import runner as RN
RUN = str(Path(__file__).resolve().parents[1] / "src" / "runner.py")

def test_rewrite_and_markers(tmp_path):
    p = RN.rewrite("torchrun --nproc_per_node=6 src/train.py --table T --out O --test_year 2020 --epochs 3"); assert p[0] == sys.executable and "--nproc_per_node=6" not in p and p[1] == "src/train.py"
    assert RN.done_marker(p) == Path("O/pred_2020.npz") and RN.done_marker(RN.rewrite("python src/ppo_manage.py --out R --test_year 2019")) == Path("R/manage_2019.json") and RN.done_marker(RN.rewrite("python src/other.py")) is None

def test_runs_jobs_concurrently_skips_done_and_reports_failures(tmp_path):
    done = tmp_path / "O"; done.mkdir(); (done / "pred_2020.npz").write_text("x"); ok = tmp_path / "ok"; ok.mkdir()
    cmds = [f"torchrun --nproc_per_node=6 src/train.py --out {done} --test_year 2020"] + [f"python -c \"import os,pathlib;pathlib.Path(r'{ok}/j{i}_'+os.environ['CUDA_VISIBLE_DEVICES']).write_text('x')\"" for i in range(4)] + ["python -c \"import sys;sys.exit(3)\""]
    f = tmp_path / "cmds.sh"; f.write_text("\n".join(cmds) + "\n"); r = subprocess.run([sys.executable, RUN, str(f), "--gpus", "0,1", "--per_gpu", "2", "--logs", str(tmp_path / "logs")], capture_output=True, text=True)
    assert r.returncode == 1 and "1 already done" in r.stdout and "1 failed" in r.stdout, r.stdout + r.stderr
    made = sorted(p.name for p in ok.iterdir()); assert len(made) == 4 and {n.split("_")[1] for n in made} == {"0", "1"}                      # jobs were spread over both GPUs
    assert "3" in (tmp_path / "logs" / "failed.txt").read_text()
