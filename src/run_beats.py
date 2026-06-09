"""BEATs challenger-lane scheduler (exp-plan-v4 S3).

Queue = BEATs (tag=beats) x folds 0-4 x seeds 41-45 = 25 runs.
Up to 3 CONCURRENT subprocesses, each pinned to ONE NPU in {4,5,6} via ASCEND_RT_VISIBLE_DEVICES.
RESUMABLE: any run whose data/oof/beats_fold{f}_seed{s}.npy exists is SKIPPED.
Each run = b1_beats_train.py single fold/seed, early-stop val macro-AUC (patience 3, cap 15),
emits per-run OOF + _rows.npy, NO checkpoints. Per-run log -> logs/run_beats_f{f}_s{s}.log.
Disk guard: abort a launch if df > MAX_DISK_PCT. Run THIS detached (setsid+nohup -> logs/beats.log).
"""
import os
import subprocess
import sys
import time

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
OOF_DIR = os.path.join(ROOT, "data", "oof")
LOG_DIR = os.path.join(ROOT, "logs")
PY = os.path.expanduser("~/SHC/miniconda3/envs/sft/bin/python")
SCRIPT = os.path.join(ROOT, "src", "b1_beats_train.py")

NPUS = [4, 5, 6]
SEEDS = [41, 42, 43, 44, 45]
FOLDS = [0, 1, 2, 3, 4]
EPOCHS = 15
PATIENCE = 3
MAX_DISK_PCT = 92
TAG = "beats"


def log(msg):
    print(f"[sched {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def disk_pct():
    out = subprocess.check_output(["df", "-P", ROOT]).decode().splitlines()[-1]
    return int(out.split()[4].rstrip("%"))


def oof_exists(f, s):
    return os.path.exists(os.path.join(OOF_DIR, f"{TAG}_fold{f}_seed{s}.npy"))


def build_queue():
    q = []
    for s in SEEDS:
        for f in FOLDS:
            if not oof_exists(f, s):
                q.append((f, s))
    return q


def launch(f, s, npu):
    runlog = os.path.join(LOG_DIR, f"run_{TAG}_f{f}_s{s}.log")
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu)
    cmd = [PY, SCRIPT, "--folds", str(f), "--seed", str(s),
           "--epochs", str(EPOCHS), "--patience", str(PATIENCE),
           "--device", "npu:0", "--tag", TAG]
    lf = open(runlog, "ab")
    lf.write(f"\n===== LAUNCH {TAG} fold{f} seed{s} on NPU{npu} @ {time.ctime()} =====\n".encode())
    lf.flush()
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    log(f"LAUNCH {TAG} fold{f} seed{s} -> NPU{npu} pid={p.pid} log={runlog}")
    return {"proc": p, "f": f, "s": s, "npu": npu, "lf": lf, "t0": time.time()}


def main():
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(OOF_DIR, exist_ok=True)
    queue = build_queue()
    total = len(FOLDS) * len(SEEDS)
    log(f"matrix {total} runs; {total - len(queue)} already have OOF (skip); "
        f"{len(queue)} to run; cards={NPUS}; disk={disk_pct()}%")
    if not queue:
        log("nothing to do, all OOF present. exit."); return

    free_npus = list(NPUS); running = []; done, failed = 0, 0
    while queue or running:
        while free_npus and queue:
            if disk_pct() > MAX_DISK_PCT:
                log(f"DISK GUARD: {disk_pct()}% > {MAX_DISK_PCT}% -> NOT launching more; "
                    f"draining {len(running)} running."); break
            f, s = queue.pop(0)
            if oof_exists(f, s):
                log(f"SKIP {TAG} fold{f} seed{s} (OOF appeared)"); continue
            npu = free_npus.pop(0)
            running.append(launch(f, s, npu))
        if not running:
            break
        time.sleep(20)
        still = []
        for r in running:
            rc = r["proc"].poll()
            if rc is None:
                still.append(r); continue
            r["lf"].close(); dt = (time.time() - r["t0"]) / 60.0
            ok = oof_exists(r["f"], r["s"])
            if rc == 0 and ok:
                done += 1
                log(f"DONE {TAG} fold{r['f']} seed{r['s']} NPU{r['npu']} rc=0 {dt:.1f}min "
                    f"OOF=yes ({done} done)")
            else:
                failed += 1
                log(f"FAIL {TAG} fold{r['f']} seed{r['s']} NPU{r['npu']} rc={rc} "
                    f"OOF={'yes' if ok else 'NO'} {dt:.1f}min -> logs/run_{TAG}_f{r['f']}_s{r['s']}.log")
            free_npus.append(r["npu"])
        running = still
    log(f"SCHEDULER EXIT. done={done} failed={failed} remaining_queue={len(queue)} disk={disk_pct()}%")


if __name__ == "__main__":
    sys.exit(main())
