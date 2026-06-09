"""Quality-max sidecar matrix scheduler — 3 concurrent NPU lanes (4,5,6).

Runs the full matrix: {cnn (author timm), pann (CNN14)} x folds 0-4 x seeds 41-45
= 50 runs, EARLY-STOP on val macro-AUC (patience 3, cap 15 epochs), reading the
decoded-waveform cache (no per-epoch ogg decode). 3 jobs run concurrently, each
pinned to ONE NPU via ASCEND_RT_VISIBLE_DEVICES in {4,5,6}; the rest queue.

Each child run emits its own per-(model,fold,seed) OOF (val rows only) +
*_rows.npy index, tagged {model}_fold{f}_seed{s}.npy (handled inside the train
scripts). The scheduler only orchestrates lanes + logging + skip-if-done.

Idempotent: a run whose {tag}_fold{f}_seed{s}.npy already exists is SKIPPED, so the
scheduler can be re-launched after an ssh drop / restart and resumes the queue.

Usage (server-side, under tmux/nohup):
  python src/b1_matrix_scheduler.py --npus 4,5,6 --seeds 41,42,43,44,45 \
      --folds 0,1,2,3,4 --models cnn,pann --epochs 15 --patience 3
"""
import argparse
import os
import subprocess
import time

LANES_DEFAULT = [4, 5, 6]


def run_spec(model):
    """Return (script, base_args) for a model tag."""
    if model == "cnn":
        return ("src/b1_cnn_train.py",
                ["--backbone", "tf_efficientnet_b0_ns", "--batch_size", "32",
                 "--num_workers", "8", "--lr", "1e-3", "--tag", "cnn"])
    if model == "pann":
        return ("src/b1_pann_train.py",
                ["--batch_size", "32", "--num_workers", "8", "--lr", "5e-4",
                 "--tag", "pann"])
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.expanduser("~/SHC/birdclef2026_clef"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--python", default=os.path.expanduser("~/SHC/miniconda3/envs/sft/bin/python"))
    ap.add_argument("--npus", default="4,5,6")
    ap.add_argument("--seeds", default="41,42,43,44,45")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--models", default="cnn,pann")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--poll", type=int, default=20, help="lane poll interval (s)")
    args = ap.parse_args()

    os.chdir(args.workdir)
    out_dir = args.out_dir or os.path.join(args.workdir, "data", "oof")
    log_dir = args.log_dir or os.path.join(args.workdir, "logs", "matrix")
    os.makedirs(log_dir, exist_ok=True)
    lanes = [int(x) for x in args.npus.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    folds = [int(x) for x in args.folds.split(",")]
    models = args.models.split(",")

    # build the job queue (model, fold, seed); skip already-done runs
    jobs = []
    skipped = 0
    for model in models:
        for f in folds:
            for s in seeds:
                done = os.path.join(out_dir, f"{model}_fold{f}_seed{s}.npy")
                if os.path.exists(done):
                    skipped += 1
                    continue
                jobs.append((model, f, s))
    total_planned = len(models) * len(folds) * len(seeds)
    print(f"[sched] planned={total_planned} skip_done={skipped} to_run={len(jobs)} "
          f"lanes={lanes} epochs={args.epochs} patience={args.patience}", flush=True)

    # lane state: npu -> (Popen, jobdesc, logfile_handle) or None
    lane_proc = {n: None for n in lanes}
    qi = 0
    n_done = 0
    n_fail = 0

    def launch(npu, job):
        model, f, s = job
        script, base = run_spec(model)
        logf = os.path.join(log_dir, f"{model}_fold{f}_seed{s}.log")
        cmd = [args.python, "-u", script,
               "--folds", str(f), "--seed", str(s),
               "--epochs", str(args.epochs), "--patience", str(args.patience),
               "--device", "npu:0"] + base
        env = dict(os.environ)
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu)  # mask to ONE NPU -> npu:0
        lh = open(logf, "w")
        lh.write(f"# CMD: ASCEND_RT_VISIBLE_DEVICES={npu} {' '.join(cmd)}\n")
        lh.flush()
        p = subprocess.Popen(cmd, stdout=lh, stderr=subprocess.STDOUT, env=env)
        print(f"[sched] LAUNCH npu{npu} <- {model} fold{f} seed{s} pid={p.pid} "
              f"log={logf}", flush=True)
        return (p, job, lh)

    while qi < len(jobs) or any(lane_proc[n] is not None for n in lanes):
        # fill idle lanes
        for npu in lanes:
            if lane_proc[npu] is None and qi < len(jobs):
                lane_proc[npu] = launch(npu, jobs[qi]); qi += 1
        time.sleep(args.poll)
        # reap finished
        for npu in lanes:
            st = lane_proc[npu]
            if st is None:
                continue
            p, job, lh = st
            rc = p.poll()
            if rc is None:
                continue
            lh.close()
            model, f, s = job
            done = os.path.join(out_dir, f"{model}_fold{f}_seed{s}.npy")
            ok = (rc == 0 and os.path.exists(done))
            if ok:
                n_done += 1
            else:
                n_fail += 1
            print(f"[sched] DONE  npu{npu} {model} fold{f} seed{s} rc={rc} "
                  f"oof_exists={os.path.exists(done)} ({'OK' if ok else 'FAIL'}) "
                  f"[done={n_done} fail={n_fail} remaining={len(jobs)-qi+sum(lane_proc[x] is not None for x in lanes)-1}]",
                  flush=True)
            lane_proc[npu] = None

    print(f"[sched] ALL FINISHED done={n_done} fail={n_fail} "
          f"(planned={total_planned} skipped_predone={skipped})", flush=True)


if __name__ == "__main__":
    main()
