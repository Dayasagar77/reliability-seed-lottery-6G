# ckpt_identity_check.py
# Decides whether the Table VII flat Dn-DQN checkpoints (ms_rel_s*.pt, run_multiseed_rel.py)
# and the Table X matched-objective Dn-DQN checkpoints (m5_dndqn_s*.pt, run_multiseed_matched.py)
# are the SAME trained models saved twice, or two INDEPENDENT 5-seed samples of one system.
#
# Must run under Windows Python (torch is not installed in the device VM).
import hashlib, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
A_FMT = 'ms_rel_s{}.pt'      # Table VII flat row
B_FMT = 'm5_dndqn_s{}.pt'    # Table X 'Proposed Dn-DQN (matched objective)' row

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def statedict_keys(ck):
    """Return the list of top-level keys whose value looks like a state_dict."""
    out = []
    for k, v in ck.items():
        if isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values()):
            out.append(k)
    return out

print('=' * 78)
print('CHECKPOINT IDENTITY CHECK  ms_rel_s*.pt  vs  m5_dndqn_s*.pt')
print('venue: torch', torch.__version__, '| dir', HERE)
print('=' * 78)

# ---- [0] structural survey on seed 0 ------------------------------------
a0 = torch.load(os.path.join(HERE, A_FMT.format(0)), map_location='cpu', weights_only=False)
b0 = torch.load(os.path.join(HERE, B_FMT.format(0)), map_location='cpu', weights_only=False)
ka, kb = sorted(a0.keys()), sorted(b0.keys())
print('\n[0] TOP-LEVEL KEYS')
print('    ms_rel_s0   :', ka)
print('    m5_dndqn_s0 :', kb)
print('    only in ms_rel   :', sorted(set(ka) - set(kb)))
print('    only in m5_dndqn :', sorted(set(kb) - set(ka)))
for k in sorted(set(ka) & set(kb)):
    va, vb = a0[k], b0[k]
    if not isinstance(va, dict):
        print('    scalar/meta  %-18s  ms_rel=%r   m5_dndqn=%r' % (k, va, vb))
for k in sorted(set(ka) - set(kb)):
    if not isinstance(a0[k], dict):
        print('    ms_rel-only  %-18s  = %r' % (k, a0[k]))
for k in sorted(set(kb) - set(ka)):
    if not isinstance(b0[k], dict):
        print('    m5-only      %-18s  = %r' % (k, b0[k]))

SD_KEYS = statedict_keys(a0)
print('    state_dict-like keys in ms_rel_s0:', SD_KEYS)
print('    state_dict-like keys in m5_s0    :', statedict_keys(b0))

# ---- [1] per-seed weight comparison -------------------------------------
print('\n[1] PER-SEED WEIGHT COMPARISON  (all state_dict-like keys present in both)')
print('    %-5s %-10s %-10s %-9s %-12s %-12s' % ('seed', 'md5_A[:8]', 'md5_B[:8]', 'n_tensors', 'all_equal', 'max|dW|'))
verdicts = []
for s in SEEDS:
    pa = os.path.join(HERE, A_FMT.format(s))
    pb = os.path.join(HERE, B_FMT.format(s))
    A = torch.load(pa, map_location='cpu', weights_only=False)
    B = torch.load(pb, map_location='cpu', weights_only=False)
    shared = [k for k in statedict_keys(A) if k in statedict_keys(B)]
    n, allq, mx = 0, True, 0.0
    for k in shared:
        sa, sb = A[k], B[k]
        if sorted(sa.keys()) != sorted(sb.keys()):
            allq = False
            print('      !! tensor-name mismatch under %r' % k)
            continue
        for t in sorted(sa.keys()):
            ta, tb = sa[t].float(), sb[t].float()
            n += 1
            if ta.shape != tb.shape:
                allq = False
                mx = float('inf')
                continue
            if not torch.equal(ta, tb):
                allq = False
            d = (ta - tb).abs().max().item()
            mx = max(mx, d)
    verdicts.append((s, allq, mx))
    print('    %-5d %-10s %-10s %-9d %-12s %-12.6g' % (
        s, md5(pa)[:8], md5(pb)[:8], n, 'IDENTICAL' if allq else 'DIFFERENT', mx))

print('\n[2] VERDICT')
if all(v[1] for v in verdicts):
    print('    ALL FIVE SEEDS IDENTICAL  ->  the two table rows are the SAME models.')
    print('    The paper must NOT print them as two rows with different numbers.')
else:
    print('    WEIGHTS DIFFER on seed(s):', [v[0] for v in verdicts if not v[1]])
    print('    min max|dW| over seeds = %.6g' % min(v[2] for v in verdicts))
    print('    -> two INDEPENDENT 5-seed training samples of one system.')
    print('    -> reportable as a replication of the seed-lottery effect (10 distinct runs).')
print('=' * 78)
