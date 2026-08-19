import sys
saved = sys.argv[:]
sys.argv = [saved[0]]
import torch
import q4_qr_branched as Q
print('module q4_qr_branched | N_QUANT =', getattr(Q, 'N_QUANT', '(absent)'))
fn = getattr(Q, 'build_qr_branched_networks', None) or getattr(Q, 'build_qr_networks', None)
if fn is None:
    cands = [n for n in dir(Q) if n.startswith('build')]
    print('builders found:', cands)
    fn = getattr(Q, cands[0]) if cands else None
print('using builder:', getattr(fn, '__name__', None))
obs = getattr(Q, 'OBS_DIM', None)
if obs is None:
    from dndqn import OBS_DIM as obs
try:
    out = fn(obs)
except TypeError:
    out = fn()
net = out[0] if isinstance(out, (tuple, list)) else out
n = sum(p.numel() for p in net.parameters())
print()
print('quantile head, N = %s : %d parameters   (Supplementary Table S29 states 65,381)  %s'
      % (getattr(Q, 'N_QUANT', '?'), n, 'ok' if n == 65381 else '*** MISMATCH ***'))
sys.argv = saved
