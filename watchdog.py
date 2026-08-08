#!/usr/bin/env python3
"""
watchdog.py -- keep the queue alive, and leave a heartbeat that proves it.

TWO FAILURE MODES, TWO LAYERS.

  process death   a stage crashes or the queue runner exits. The watchdog notices within one
                  tick and restarts the runner. This layer works.

  container reclaim   the whole sandbox is torn down -- the watchdog dies with everything else,
                  so it cannot restart anything. Nothing INSIDE the container can survive that.
                  What it CAN do is make the outage self-evident afterwards: the heartbeat file
                  stops advancing, and its age on the next wake is exactly how long was lost.
                  The outer layer is a scheduled wake-up that brings the session back, and the
                  first thing to do on waking is read the heartbeat age.

PROGRESS IS READ FROM DISK, NOT REPORTED BY THE STAGES. A stage that has died cannot tell you it
died, and one that is wedged will happily keep claiming it is fine. Row counts in the result CSVs
and .done markers next to the checkpoints are ground truth: they only move when real work lands.
"""
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
HB = os.path.join(HERE, 'heartbeat.json')
TICK = 30


def rows(path):
    try:
        with open(path) as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def progress():
    p = {
        'M4_robustness_evals': [rows('results/gsweep/gsweep.csv'), 240],
        'Q1_latency_pairs': [rows('results/q1latency/q1_raw.csv'), 40],
        'M2_sensitivity_evals': [rows('results/m2sweep/m2sweep.csv'), 200],
        'fixedseed_units': [len([f for f in os.listdir('results/fixedseed')
                                 if f.startswith('unit_')]) if os.path.isdir('results/fixedseed')
                            else 0, 5],
        'varyseed_units': [len([f for f in os.listdir('results/varyseed')
                                if f.startswith('unit_')]) if os.path.isdir('results/varyseed')
                           else 0, 5],
        'trainings_done': [len([f for f in os.listdir('.') if f.endswith('.pt.done')]), 10],
    }
    return p


def alive(pattern):
    try:
        out = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def queue_pending():
    try:
        q = [l.strip() for l in open('queue.txt')
             if l.strip() and not l.strip().startswith('#')]
    except OSError:
        return 0
    try:
        d = set(l.strip() for l in open('queue_done.txt'))
    except OSError:
        d = set()
    return len([c for c in q if c not in d])


def main():
    restarts = 0
    started = time.time()
    while True:
        pend = queue_pending()
        run_alive = alive('[q]ueue_runner.py')
        if pend and not run_alive:
            subprocess.Popen(['nohup', 'python3', '-u', 'queue_runner.py'],
                             stdout=open('queue.log', 'a'), stderr=subprocess.STDOUT,
                             start_new_session=True)
            restarts += 1
            time.sleep(3)
            run_alive = alive('[q]ueue_runner.py')
        hb = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'epoch': int(time.time()),
            'watchdog_uptime_min': round((time.time() - started) / 60, 1),
            'queue_runner_alive': run_alive,
            'queue_stages_pending': pend,
            'runner_restarts_by_watchdog': restarts,
            'progress': progress(),
        }
        tmp = HB + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(hb, f, indent=1)
        os.replace(tmp, HB)
        time.sleep(TICK)


if __name__ == '__main__':
    main()
