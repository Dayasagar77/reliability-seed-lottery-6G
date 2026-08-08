#!/usr/bin/env python3
"""
queue_runner.py -- run the campaign queue in order, resumably.

Replaces a chain of `while pgrep …; do sleep; done` waiters. That pattern already produced one
silent deadlock (the waiter matched its own command line) and it does not compose: six chained
links is six chances to hang, and inserting a stage in the middle means rebuilding the chain.

This reads queue.txt each pass and runs the first line not yet recorded in queue_done.txt, so a
stage can be INSERTED ahead of a pending one at any time simply by editing the file. Finished
stages are never re-run. If the queue empties it idles and keeps re-reading, so later work can be
appended without restarting anything; it gives up after IDLE_EXIT minutes of nothing to do.
"""
import os
import subprocess
import sys
import time

Q, DONE = 'queue.txt', 'queue_done.txt'
IDLE_EXIT = 90


def log(m):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' | queue | ' + m, flush=True)


def read(p):
    if not os.path.exists(p):
        return []
    return [l.strip() for l in open(p) if l.strip() and not l.strip().startswith('#')]


idle = 0
while True:
    todo = [c for c in read(Q) if c not in read(DONE)]
    if not todo:
        idle += 1
        if idle * 0.5 >= IDLE_EXIT:
            log('idle %d min with an empty queue — exiting' % IDLE_EXIT)
            break
        time.sleep(30)
        continue
    idle = 0
    cmd = todo[0]
    log('START  %s   (%d stage(s) pending)' % (cmd, len(todo)))
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True)
    log('FINISH %s   rc=%d in %.1f min' % (cmd, rc, (time.time() - t0) / 60))
    with open(DONE, 'a') as f:
        f.write(cmd + '\n')
    if rc != 0:
        log('*** stage returned %d — continuing to the next stage, but this needs eyes ***' % rc)
