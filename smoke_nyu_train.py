#!/usr/bin/env python3
"""Smoke test: run the NYU trainer for a few thousand steps to confirm it trains without
error and to measure throughput (steps/sec) so we can estimate the full 800k-step ETA."""
import time, train_phase2_nyu  # sets t2.Phase2Env_v6 = Phase2Env_NYU, save path
import train_phase2_v6 as t2

t2.TOTAL_TIMESTEPS = 4000
t2.LEARNING_STARTS = 500
t2.LOG_INTERVAL    = 1000
t2.MODEL_SAVE_PATH = "tmp_smoke_nyu.pt"

t0 = time.time()
t2.main()
dt = time.time() - t0
print(f"\nSMOKE OK: {t2.TOTAL_TIMESTEPS} steps in {dt:.1f}s -> {t2.TOTAL_TIMESTEPS/dt:.0f} steps/s")
print(f"Estimated full 800k-step ETA: {800_000/(t2.TOTAL_TIMESTEPS/dt)/60:.1f} min")
