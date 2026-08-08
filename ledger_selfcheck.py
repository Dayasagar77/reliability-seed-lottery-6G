#!/usr/bin/env python3
"""ledger_selfcheck.py -- READ-ONLY.  Mechanise §8 rule 3 against the ledger itself.

§8 rule 3 says: "Every edit that reaches the docx has a line in this ledger.  An edit without a
ledger line is a defect, whether or not the value is right."  Until now that rule has been checked
by reading, and reading is exactly how [93], [95], [116] and [132] stayed unregistered through
several passes -- and how a bracket-only grep then reported six MORE blocks as unregistered when
in fact they were registered under the bare `block N` form.  Both failures are grep failures, and
the fix for a grep failure is not a more careful grep, it is a checker that knows both forms.

WHAT THIS CHECKS
  1.  Every block the §7 OD-2 grouping claims exists in the manuscript at the pinned md5, and its
      TYPE (table vs paragraph) matches how the ledger refers to it -- the ledger's convention is
      bare `96` for tables, bracketed `[97]` for prose, and a type mismatch means the convention
      has drifted, which is how a caption gets edited as if it were a table.
  2.  Every block in the OD-2 grouping has at least one registration line somewhere in §§1-5,
      counting BOTH reference forms.  A block named as work-to-do with no line describing the work
      is rule 3's defect, stated in the ledger's own terms.
  3.  Every bracketed integer in the ledger that is <= 276 but has NO registration line anywhere is
      listed with context, so citations quoted from the manuscript ("[4]", "[38]") can be told
      apart from block references by eye -- the one judgement that cannot be automated is left to
      a human, but the candidate list is generated mechanically so it cannot be short.

It refuses on md5 drift and writes nothing but stdout.
"""
import hashlib
import re
from collections import defaultdict

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

DOCX = '/home/claude/ms/DnDQN_manuscript_updated_IXrenumbered.docx'
EXPECT = 'ddea24ec8ad709ab02356486b18c46e2'
LEDGER = '/home/claude/ledger/DnDQN_Paper1_ManuscriptChangeLedger_20260802.md'

# The §7 OD-2 grouping, transcribed.  Tables are bare in the ledger, prose is bracketed; the
# expected type is recorded here so the checker can catch a convention drift rather than trust it.
OD2 = {
    5: ('para', 'OD-2d  abstract'),
    13: ('para', 'OD-2c  contribution 3'),
    93: ('para', 'OD-2b  §V-B heading'),
    95: ('para', 'OD-2a  Table IV caption'),
    96: ('table', 'OD-2a  Table IV'),
    97: ('para', 'OD-2a  prose'),
    101: ('para', 'OD-2a  Fig. 7 caption (also OD-4)'),
    116: ('para', 'OD-2a  Table V caption'),
    117: ('table', 'OD-2a  Table V'),
    132: ('para', 'OD-2e  stack and protocol'),
    133: ('para', 'OD-2e  baselines'),
    140: ('para', 'OD-2a  Table VII caption'),
    141: ('table', 'OD-2a  Table VII'),
    143: ('para', 'OD-2c  finding 1'),
    144: ('para', 'OD-2c  finding 2'),
    161: ('para', 'OD-2a  Table X caption'),
    162: ('table', 'OD-2a  Table X'),
    163: ('para', 'OD-2a  prose'),
    194: ('para', 'OD-2a  prose'),
    207: ('para', 'OD-2e  threats to validity'),
    209: ('para', 'OD-2b  conclusion'),
    94: ('para', 'verify-only  §V-B lead-in'),
}


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def iter_block_items(parent):
    for child in parent.element.body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


got = md5(DOCX)
print(f'manuscript md5 {got}  {"OK" if got == EXPECT else "DRIFT -- REFUSING"}')
if got != EXPECT:
    raise SystemExit(1)
blocks = list(iter_block_items(Document(DOCX)))
print(f'manuscript: {len(blocks)} blocks')

lines = open(LEDGER, encoding='utf-8').read().split('\n')
print(f'ledger    : {len(lines)} lines, md5 {md5(LEDGER)}\n')

# Where does §5 (and §§1-4) start and end?  Registration lines are everything before §6.
sec6 = next((i for i, L in enumerate(lines) if L.startswith('## 6.')), len(lines))
reg_zone = lines[:sec6]

# Both reference forms, per line, restricted to the registration zone.
brk = re.compile(r'\[(\d{1,3})\]')
bare = re.compile(r'(?<![\d.])(?:block|blocks|Block)\s+(\d{1,3})\b|\*\*(\d{1,3})\*\*|^\|\s*(\d{1,3})\s*\(')

registered = defaultdict(list)
for i, L in enumerate(reg_zone):
    for m in brk.finditer(L):
        registered[int(m.group(1))].append(i + 1)
    for m in bare.finditer(L):
        n = next(g for g in m.groups() if g)
        registered[int(n)].append(i + 1)

print('=' * 118)
print('  CHECK 1 + 2 -- the OD-2 grouping: does each block exist, is its type right, is it registered?')
print('=' * 118)
print(f'{"blk":>5} {"type":>7} {"expected":>8}  {"reg?":>5}  role / first text')
print('-' * 118)
bad_type, unreg = [], []
for n in sorted(OD2):
    role_type, role = OD2[n]
    if n >= len(blocks):
        print(f'{n:>5}  OUT OF RANGE -- manuscript has {len(blocks)} blocks')
        bad_type.append(n)
        continue
    b = blocks[n]
    actual = 'table' if isinstance(b, Table) else 'para'
    if actual == 'table':
        txt = 'TABLE ' + f'{len(b.rows)}x{len(b.columns)}  ' + \
              ' | '.join(c.text.strip()[:14] for c in b.rows[0].cells)
    else:
        txt = b.text.strip().replace('\n', ' ')[:58]
    ok_type = '' if actual == role_type else '  <-- TYPE MISMATCH'
    if ok_type:
        bad_type.append(n)
    hits = registered.get(n, [])
    if not hits:
        unreg.append(n)
    flag = f'{len(hits):>5}' if hits else '  NONE'
    print(f'{n:>5} {actual:>7} {role_type:>8}  {flag}  {role} :: {txt}{ok_type}')

print()
print(f'  type mismatches : {bad_type if bad_type else "none"}')
print(f'  UNREGISTERED    : {unreg if unreg else "none -- rule 3 satisfied for the OD-2 grouping"}')

print()
print('=' * 118)
print('  CHECK 3 -- bracketed integers <= 276 with no registration line anywhere in §§0-5')
print('           (citations quoted from the manuscript live here too; adjudicate by eye)')
print('=' * 118)
all_brk = defaultdict(list)
for i, L in enumerate(lines):
    for m in brk.finditer(L):
        all_brk[int(m.group(1))].append((i + 1, L.strip()))
orphans = sorted(n for n in all_brk if n < len(blocks) and n not in registered)
for n in orphans:
    ln, L = all_brk[n][0]
    print(f'\n  [{n}]  {len(all_brk[n])} mention(s), first at ledger line {ln}')
    print(f'        ledger: {L[:150]}')
    print(f'        docx  : {blocks[n].text.strip()[:110] if isinstance(blocks[n], Paragraph) else "TABLE"}')
if not orphans:
    print('\n  none')

print('\n' + '=' * 118)
print(f'  VERDICT: {"PASS" if not bad_type and not unreg else "FAIL"}'
      f'   (type mismatches {len(bad_type)}, unregistered {len(unreg)})')
print('=' * 118)
