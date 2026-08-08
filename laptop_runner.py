#!/usr/bin/env python3
"""
laptop_runner.py — runs the 40-training revision campaign on the author's Windows laptop,
N units at a time, resumably, with a live-adjustable slot count and a hard RAM floor.

WHY THE LAPTOP AND NOT THE CLOUD
--------------------------------
The two venues do not run the same toolchain:

    laptop   Python 3.11.9   torch 2.5.1+cu121    numpy 2.4.4
    cloud    Python 3.11.15  torch 2.13.0+cu130   numpy 2.4.4

The same seed, the same config and the same 800k steps give measurably different policies across
them (seed 0 flat: HO 93.00 / Pwr 18.96 / Tput 485.9 on the laptop against HO 91.04 / Pwr 24.52 /
Tput 473.7 in the cloud). Every number already published in the manuscript came off torch 2.5.1.
Retraining the baselines on torch 2.13 would put two toolchains inside one comparison table — a
milder form of the exact inconsistency the Results in Engineering AE rejected the paper for. So the
reporting pipeline is the laptop, and the cloud campaign keeps running as an INDEPENDENT
cross-toolchain replication arm. Both are labelled in the CSVs via PIPELINE_TAG; neither is pooled
into the other.

WHAT THIS DOES NOT CHANGE
-------------------------
Nothing about any training. Each unit is one ordinary driver invocation for one seed. train() calls
set_seed(seed) and builds fresh networks, so a seed's policy does not depend on how many siblings
shared the machine. Every process is pinned to ONE thread (OMP/MKL=1 plus torch.set_num_threads(1)
inside each driver), which is the paper's stated protocol, so parallelism here is N independent
single-threaded trainings on N cores — not a change of compute model.

  RAM is the binding constraint, not cores. Each training holds ~743 MB resident. The runner
  refuses to start a new unit while available memory is under --floor-mb, and it re-reads
  campaign_slots.txt every poll, so the width can be raised or lowered WHILE THE CAMPAIGN RUNS
  without killing anything.

Usage (PowerShell, from the folder holding the drivers):
    python -u laptop_runner.py --slots 5
    python -u laptop_runner.py --slots 5 --dry-run       # print the schedule, launch nothing
    "6" | Out-File -Encoding ascii campaign_slots.txt    # widen to 6 while it runs
    Get-Content -Wait campaign_campaign.log              # follow along

Resume: rerun the same command. A unit is complete when its own CSV holds its row, so an
interrupted campaign restarts only what was actually in flight.
"""
import argparse, csv, os, subprocess, sys, time

import campaign_units as CU

LOG = 'laptop_campaign.log'
STATUS = 'laptop_campaign.status'
SLOTS_FILE = 'campaign_slots.txt'


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# --------------------------------------------------------------------------------------------
# Memory. Windows via GlobalMemoryStatusEx, Linux via /proc/meminfo, so the runner can be
# rehearsed in the cloud container before it is trusted with the real campaign.
# --------------------------------------------------------------------------------------------
def avail_mb():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys // (1024 * 1024)
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 1 << 30


def keep_awake(on=True):
    """Stop Windows sleeping mid-campaign. Process-scoped: it lapses when this process exits, so
    it cannot leave the machine permanently unable to sleep. No admin rights needed."""
    if os.name != 'nt':
        return False
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
    flags = (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED) if on else ES_CONTINUOUS
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


def rows_in(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, newline='') as f:
            return sum(1 for _ in csv.DictReader(f))
    except OSError:
        return 0


def want_slots(default):
    """Re-read every poll so the width is adjustable mid-campaign without a restart."""
    try:
        v = int(open(SLOTS_FILE).read().strip())
        return max(1, min(40, v))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slots', type=int, default=5, help='concurrent trainings (start conservative)')
    ap.add_argument('--floor-mb', type=int, default=1400, help='never launch below this free RAM')
    ap.add_argument('--venue', default='laptop')
    ap.add_argument('--poll', type=int, default=20)
    ap.add_argument('--max-restarts', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    units = CU.units(venue=a.venue)
    if not os.path.exists(SLOTS_FILE):
        with open(SLOTS_FILE, 'w') as f:
            f.write(str(a.slots) + '\n')

    log("=" * 96)
    log(f"CAMPAIGN START | {len(units)} units | venue={a.venue} | slots={a.slots} "
        f"(live via {SLOTS_FILE}) | RAM floor {a.floor_mb} MB | avail now {avail_mb()} MB")
    log(f"python {sys.version.split()[0]} | cwd {os.getcwd()}")
    try:
        import torch, numpy
        log(f"toolchain | torch {torch.__version__} | numpy {numpy.__version__}")
    except Exception as e:                                   # noqa: BLE001 — advisory only
        log(f"toolchain | could not import torch/numpy in the runner: {e}")

    done = {u['name'] for u in units if rows_in(u['rows_path']) >= 1}
    log(f"resume | {len(done)} of {len(units)} units already have their row: "
        f"{sorted(done) if done else 'none'}")

    if a.dry_run:
        for u in units:
            mark = 'DONE' if u['name'] in done else '    '
            log(f"  {mark} {u['est_min']:3d}m {u['name']:<22} {u['script']} {' '.join(u['argv'])}")
        log("DRY RUN — nothing launched")
        return

    if keep_awake(True):
        log("power | sleep inhibited for the duration of this process")

    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    env['PIPELINE_TAG'] = f'{a.venue}-singlepass'
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')

    running = {}                       # name -> (Popen, started_at, logfile handle)
    restarts = {}
    queue = [u for u in units if u['name'] not in done]
    t0 = time.time()

    try:
        while queue or running:
            # ---- reap ----
            for name in list(running):
                p, started, fh = running[name]
                if p.poll() is None:
                    continue
                fh.close()
                u = next(x for x in units if x['name'] == name)
                mins = (time.time() - started) / 60
                if p.returncode == 0 and rows_in(u['rows_path']) >= 1:
                    done.add(name)
                    log(f"DONE  {name} rc=0 in {mins:.1f}m -> {u['rows_path']}")
                else:
                    n = restarts.get(name, 0) + 1
                    restarts[name] = n
                    if n <= a.max_restarts:
                        log(f"FAIL  {name} rc={p.returncode} after {mins:.1f}m, no row written — "
                            f"requeue (restart {n}/{a.max_restarts}); see logs/{name}.log")
                        queue.append(u)
                    else:
                        log(f"GIVEUP {name} rc={p.returncode} after {a.max_restarts} restarts — "
                            f"NOT retrying; investigate logs/{name}.log before merging")
                del running[name]

            # ---- launch ----
            slots = want_slots(a.slots)
            while queue and len(running) < slots:
                free = avail_mb()
                if free < a.floor_mb:
                    log(f"HOLD  {free} MB free < floor {a.floor_mb} MB — not starting a new unit "
                        f"({len(running)} running, {len(queue)} queued)")
                    break
                u = queue.pop(0)
                fh = open(os.path.join('logs', u['name'] + '.log'), 'a', encoding='utf-8')
                fh.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                fh.flush()
                p = subprocess.Popen([sys.executable, '-u', u['script']] + u['argv'],
                                     stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
                running[u['name']] = (p, time.time(), fh)
                log(f"START {u['name']:<22} pid {p.pid} | slot {len(running)}/{slots} | "
                    f"{free} MB free | ~{u['est_min']}m single-job")

            nd, nr, nq = len(done), len(running), len(queue)
            with open(STATUS, 'w') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} done={nd}/{len(units)} "
                        f"running={nr} queued={nq} slots={slots} free_mb={avail_mb()} "
                        f"elapsed_min={(time.time() - t0) / 60:.1f}\n")
                for name, (p, started, _) in sorted(running.items()):
                    f.write(f"  {name} pid {p.pid} {(time.time() - started) / 60:.1f}m\n")
            if not queue and not running:
                break
            time.sleep(a.poll)

        el = (time.time() - t0) / 60
        log(f"CAMPAIGN COMPLETE | {len(done)}/{len(units)} units carry a row | {el:.1f} min "
            f"({el / 60:.2f} h) wall")
        missing = [u['name'] for u in units if u['name'] not in done]
        if missing:
            log(f"INCOMPLETE — no row for: {missing}. Merge will refuse until these are resolved.")
        else:
            log("next: python -u merge_shards.py --venue " + a.venue)
    except KeyboardInterrupt:
        log(f"INTERRUPTED — {len(running)} unit(s) still running as child processes: "
            f"{sorted(running)}. They are independent; rerun this script to resume the rest.")
    finally:
        keep_awake(False)


if __name__ == '__main__':
    main()
