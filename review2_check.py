#!/usr/bin/env python3
"""review2_check.py -- verify the second reviewer's checkable factual claims against v27.

A review is evidence, not instruction. Several of these claims are arithmetic or presence checks
that the manuscript can settle, and acting on a claim that is already addressed -- or that is
wrong -- wastes the revision. Each item below prints what the document actually says.
"""
import re
import unicodedata
import zipfile

DOC = 'DnDQN_manuscript_v27_M3_20260803.docx'
raw = zipfile.ZipFile(DOC).read('word/document.xml').decode('utf8')
T = re.sub(r'<[^>]+>', '', raw)


def norm(s):
    s = unicodedata.normalize('NFKD', s)
    for a, b in (('−', '-'), ('–', '-'), ('—', '-'), (' ', ' '), (' ', ' '), ('×', 'x')):
        s = s.replace(a, b)
    return s


N = norm(T)


def show(label, pat, n=3, width=200):
    print('\n--- %s' % label)
    hits = list(re.finditer(pat, N, re.I))
    if not hits:
        print('    NO MATCH')
        return
    for m in hits[:n]:
        print('    ...%s...' % N[max(0, m.start() - width // 2):m.end() + width].replace('\n', ' '))
    if len(hits) > n:
        print('    (+%d more)' % (len(hits) - n))


print('=' * 100)
print('SECOND REVIEW -- CHECKABLE CLAIMS AGAINST %s' % DOC)
print('=' * 100)

print('\n[M2] the 22.6x ratio and its components')
for v in ('22.6', '22.7', '20.70', '0.91'):
    print('    %-6s occurs %d time(s)' % (v, len(re.findall(r'(?<![\d.])' + re.escape(v) + r'(?![\d])', N))))
print('    20.70 / 0.91 = %.3f   (reviewer says the abstract rounds to 22.6, arithmetic gives 22.75)'
      % (20.70 / 0.91))

print('\n[M5] training-budget figures')
for v in (r'8x10', r'5x10', r'8\s*x\s*10', r'800,000', r'500,000'):
    print('    %-12s occurs %d time(s)' % (v, len(re.findall(v, N, re.I))))
show('budget mentions in context', r'\dx10⁵|\dx10\^?5', 4, 160)

print('\n[C2 / M10] Holm correction coverage')
for v in ('Holm', 'Bonferroni', 'family-wise', 'multiple compar'):
    print('    %-16s occurs %d time(s)' % (v, len(re.findall(v, N, re.I))))
show('Holm in context', r'Holm', 3, 220)

print('\n[C4] the BDQ / branched-incremental regenerated checkpoints')
show('regenerated checkpoints', r'regenerat', 3, 260)

print('\n[T9] latency hardware')
show('latency hardware', r'Xeon|i9-12900H|2-core', 4, 200)

print('\n[M7] biography placeholders')
print('    PLACEHOLDER occurs %d time(s)' % len(re.findall(r'PLACEHOLDER', N, re.I)))

print('\n[C6 / rec 4] is the n-step / fading material already relocated?')
for probe in ('APPENDIX', 'given in Table XXII, Appendix A', 'given in Table XXIII, Appendix B'):
    print('    %-40s occurs %d time(s)' % (probe, N.count(norm(probe))))

print('\n[M9] the ping-pong claim')
show('ping-pong', r'ping-?pong', 3, 220)

print('\n[Q5 / N310] N310 discussion')
show('N310', r'N310\s*=\s*\d', 4, 180)
print('\n' + '=' * 100)
