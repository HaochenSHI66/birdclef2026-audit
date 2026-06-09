"""Sidecar training-matrix scheduler.

Queue = {author-CNN (tag=cnn), PANNs CNN14 (tag=pann)} x folds 0-4 x seeds 41-45
      = 2 * 5 * 5 = 50 runs.

Runs up to 3 CONCURRENT subprocesses, each pinned to ONE NPU (4, 5, 6) via
ASCEND_RT_VISIBLE_DEVICES. As each finishes, the next queued run is launched on
the freed card. RESUMABLE: any run whose per-run OOF
  data/oof/{tag}_fold{f}_seed{s}.npy
already exists is SKIPPED.

Each run invokes b1_cnn_train.py / b1_pann_train.py with a SINGLE fold and SINGLE
seed, early-stop on val macro-AUC (patience 3, cap 15 epochs). The training scripts
already: emit per-run OOF + _rows.npy, log val-macro-AUC per epoch, mmap the wav
cache, and save NO model checkpoints. Each run's stdout/stderr -> its own per-run
log logs/run_{tag}_f{f}_s{s}.log .

Disk guard: before launching any run, if df usage > MAX_DISK_PCT abort that launch
(and the scheduler) rather than risk filling the disk.

Run THIS scheduler itself detached (setsid+nohup -> logs/sidecars.log).
"""
import os
import subprocess
import sys
import time

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
OOF_DIR = os.path.join(ROOT, "data", "oof")
LOG_DIR = os.path.join(ROOT, "logs")
PY = os.path.expanduser("~/SHC/miniconda3/envs/sft/bin/python")

NPUS = [4, 5, 6]            # ONLY these cards
SEEDS = [41, 42, 43, 44, 45]
FOLDS = [0, 1, 2, 3, 4]
EPOCHS = 15                 # cap; early-stop patience 3 ends most sooner
PATIENCE = 3
MAX_DISK_PCT = 92          # abort a launch if disk strictly above this

SCRIPTS = {
    "cnn": os.path.join(ROOT, "src", "b1_cnn_train.py"),
    "pann": os.path.join(ROOT, "src", "b1_pann_train.py"),
}
# author-CNN first so its (untested-with-cache) timing surfaces in the first wave,
# PANN second; interleave so the opening 3-wide wave covers both model families.
TAGS = ["cnn", "pann"]


def log(msg):
    print(f"[sched {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def disk_pct():
    out = subprocess.check_output(["df", "-P", ROOT]).decode().splitlines()[-1]
    return int(out.split()[4].rstrip("%"))


def oof_exists(tag, f, s):
    return os.path.exists(os.path.join(OOF_DIR, f"{tag}_fold{f}_seed{s}.npy"))


def build_queue():
    q = []
    # iterate seeds-outer so early runs span folds (varied timing), tags interleaved
    for s in SEEDS:
        for f in FOLDS:
            for tag in TAGS:
                if not oof_exists(tag, f, s):
                    q.append((tag, f, s))
    return q


def launch(tag, f, s, npu):
    script = SCRIPTS[tag]
    runlog = os.path.join(LOG_DIR, f"run_{tag}_f{f}_s{s}.log")
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu)
    cmd = [
        PY, script,
        "--folds", str(f),
        "--seed", str(s),
        "--epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--device", "npu:0",   # logical 0 within the pinned card
        "--tag", tag,
    ]
    lf = open(runlog, "ab")
    lf.write(f"\n===== LAUNCH {tag} fold{f} seed{s} on NPU{npu} @ {time.ctime()} =====\n".encode())
    lf.flush()
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    log(f"LAUNCH {tag} fold{f} seed{s} -> NPU{npu} pid={p.pid} log={runlog}")
    return {"proc": p, "tag": tag, "f": f, "s": s, "npu": npu, "lf": lf,
            "t0": time.time()}


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    queue = build_queue()
    total_target = 2 * len(FOLDS) * len(SEEDS)
    log(f"matrix {total_target} runs; {total_target - len(queue)} already have OOF (skip); "
        f"{len(queue)} to run; cards={NPUS}; disk={disk_pct()}%")
    if not queue:
        log("nothing to do, all OOF present. exit.")
        return

    free_npus = list(NPUS)
    running = []   # list of run dicts
    done, failed = 0, 0

    while queue or running:
        # launch onto every free card while queue non-empty
        while free_npus and queue:
            if disk_pct() > MAX_DISK_PCT:
                log(f"DISK GUARD: {disk_pct()}% > {MAX_DISK_PCT}% -> NOT launching more; "
                    f"draining {len(running)} running.")
                break
            tag, f, s = queue.pop(0)
            if oof_exists(tag, f, s):       # double-check (resume race)
                log(f"SKIP {tag} fold{f} seed{s} (OOF appeared)")
                continue
            npu = free_npus.pop(0)
            running.append(launch(tag, f, s, npu))

        if not running:
            # queue non-empty but blocked (disk guard) -> stop
            break

        # poll running set
        time.sleep(20)
        still = []
        for r in running:
            rc = r["proc"].poll()
            if rc is None:
                still.append(r)
                continue
            r["lf"].close()
            dt = (time.time() - r["t0"]) / 60.0
            ok = oof_exists(r["tag"], r["f"], r["s"])
            if rc == 0 and ok:
                done += 1
                log(f"DONE {r['tag']} fold{r['f']} seed{r['s']} NPU{r['npu']} "
                    f"rc=0 {dt:.1f}min OOF=yes ({done} done)")
            else:
                failed += 1
                log(f"FAIL {r['tag']} fold{r['f']} seed{r['s']} NPU{r['npu']} "
                    f"rc={rc} OOF={'yes' if ok else 'NO'} {dt:.1f}min "
                    f"-> see logs/run_{r['tag']}_f{r['f']}_s{r['s']}.log")
            free_npus.append(r["npu"])
        running = still

    log(f"SCHEDULER EXIT. done={done} failed={failed} remaining_queue={len(queue)} "
        f"disk={disk_pct()}%")


if __name__ == "__main__":
    sys.exit(main())
