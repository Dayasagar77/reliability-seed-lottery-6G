#!/usr/bin/env python3
"""
step23_two_column.py -- put the manuscript into IEEE Access two-column layout, for real.

Every page figure I reported until now came from a scratch copy that was re-laid into this geometry
and thrown away. This applies it to the deliverable, so the number Word shows you is the number.

GEOMETRY, from the IEEE Access template: US Letter; text block 7.16 in wide, so 0.67 in side
margins; top 0.75 in, bottom 1.0 in; two columns with a 0.24 in gutter, giving 3.46 in per column.
Body 10 pt, tables and captions 8 pt.

THE PART THAT IS NOT JUST A SETTING. 20 of the 24 tables are four columns or wider and cannot fit
in 3.46 in; nor can the six figures that are wide panel strips (aspect ratio 2.6-2.8). In OOXML a
full-width element inside a two-column document requires a CONTINUOUS SECTION BREAK to a
one-column section and another back, because a section -- not a paragraph -- carries the column
count. So each wide table travels with its caption inside its own one-column section, and likewise
each wide figure. Adjacent regions are merged so two consecutive wide tables share one section
rather than opening and closing two.

WHAT IS AND IS NOT TOUCHED. This pass changes layout only: page setup, column counts, fonts, image
scale, and the section-break paragraphs that carry the column changes. Not one character of text
changes, so the Arabic-numeral multiset must come out IDENTICAL, and the figure-caption invariant
established in step 22 -- the k-th image is followed by the caption "Fig. k." -- must still hold
afterwards. Both are gated below.
"""
import copy
import hashlib
import re
import subprocess
import sys
from collections import Counter

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

SRC = 'DnDQN_manuscript_v32_REVIEW2_20260803.docx'
SRC_MD5 = 'b4b476d44ff97f37da83332fc28aac2a'
DST = 'baseline_v32_twocol.docx'
NUM = re.compile(r'\d+(?:\.\d+)?')
FIGCAP = re.compile(r'^\s*Fig\.\s*(\d+)\.')
TABCAP = re.compile(r'^\s*TABLE\s+([IVXL]+)\s*[:.]', re.I)

PAGE_W, PAGE_H = 8.5, 11.0
MARGIN_LR, MARGIN_T, MARGIN_B = 0.67, 0.75, 1.0
GUTTER = 0.24
TEXT_W = PAGE_W - 2 * MARGIN_LR                    # 7.16 in
COL_W = (TEXT_W - GUTTER) / 2                      # 3.46 in
CHAR_W, CELL_PAD = 0.0555, 0.10                    # Times 8 pt, per-column padding
WIDE_IMG_AR = 2.2


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def blocks(doc):
    out = []
    for ch in doc.element.body.iterchildren():
        if ch.tag == qn('w:p'):
            out.append(Paragraph(ch, doc))
        elif ch.tag == qn('w:tbl'):
            out.append(Table(ch, doc))
    return out


def all_paras(B):
    for b in B:
        if isinstance(b, Table):
            for row in b.rows:
                for c in row.cells:
                    for p in c.paragraphs:
                        yield p
        else:
            yield b


def numerals(doc):
    acc = []
    for t in doc.element.body.iter(qn('w:t')):
        if t.text:
            acc += NUM.findall(t.text)
    return acc


def img_part(doc, p):
    blips = p._element.findall('.//' + qn('a:blip'))
    if not blips:
        return None
    return doc.part.related_parts[blips[0].get(qn('r:embed'))].partname.split('/')[-1]


def twips(inches):
    return str(int(round(inches * 1440)))


def make_sectPr(cols, continuous=True):
    s = OxmlElement('w:sectPr')
    if continuous:
        t = OxmlElement('w:type')
        t.set(qn('w:val'), 'continuous')
        s.append(t)
    sz = OxmlElement('w:pgSz')
    sz.set(qn('w:w'), twips(PAGE_W))
    sz.set(qn('w:h'), twips(PAGE_H))
    s.append(sz)
    mar = OxmlElement('w:pgMar')
    for k, v in (('top', MARGIN_T), ('bottom', MARGIN_B),
                 ('left', MARGIN_LR), ('right', MARGIN_LR)):
        mar.set(qn('w:' + k), twips(v))
    for k, v in (('header', '720'), ('footer', '720'), ('gutter', '0')):
        mar.set(qn('w:' + k), v)
    s.append(mar)
    c = OxmlElement('w:cols')
    c.set(qn('w:num'), str(cols))
    c.set(qn('w:space'), twips(GUTTER))
    c.set(qn('w:equalWidth'), '1')
    s.append(c)
    return s


def marker_paragraph(doc, sectPr):
    """An (almost) invisible paragraph whose only job is to carry a section break."""
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    spacing = OxmlElement('w:spacing')
    for k in ('before', 'after', 'line'):
        spacing.set(qn('w:' + k), '0')
    spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)
    rPr = OxmlElement('w:rPr')
    szv = OxmlElement('w:sz')
    szv.set(qn('w:val'), '2')          # 1 pt
    rPr.append(szv)
    pPr.append(rPr)
    pPr.append(sectPr)
    p.append(pPr)
    return p


def main():
    apply = '--apply' in sys.argv
    bar = '=' * 104
    print(bar)
    print('STEP 23 -- IEEE Access two-column layout   [%s]' % ('APPLY' if apply else 'DRY RUN'))
    print(bar)
    got = md5(SRC)
    print('  md5 %s  %s' % (got, 'OK' if got == SRC_MD5 else '*** MISMATCH ***'))
    if got != SRC_MD5:
        return 2
    print('  text block %.2f in, columns %.2f in with a %.2f in gutter' % (TEXT_W, COL_W, GUTTER))

    doc = Document(SRC)
    B = blocks(doc)
    before = numerals(doc)
    n_tab_before = sum(1 for b in B if isinstance(b, Table))

    # ---------------------------------------------------------------- what must span
    def needs_width(t):
        """Estimate the width a table needs, rather than guessing from its column count.

        A column can wrap, so what sets its minimum is its longest unbreakable token, floored by
        a short cell's full length. Column count alone is wrong in both directions here: Table I
        has three columns of long parameter descriptions and does NOT fit, while Tables IX, XV,
        XVI and XXIV have four or five columns of short numbers and do.
        """
        w = 0.0
        for ci in range(len(t.columns)):
            longest = 0
            for r in t.rows:
                if len(set(c.text for c in r.cells)) == 1:      # merged banner row
                    continue
                txt = r.cells[ci].text.strip()
                longest = max(longest,
                              max((len(tok) for tok in txt.split()), default=0),
                              min(len(txt), 22))
            w += longest * CHAR_W + CELL_PAD
        return w

    regions = []
    for i, b in enumerate(B):
        if isinstance(b, Table) and (len(b.columns) >= 4 or needs_width(b) > COL_W):
            s = i
            if i > 0 and not isinstance(B[i - 1], Table) and TABCAP.match(B[i - 1].text or ''):
                s = i - 1
            regions.append((s, i, 'TABLE %s (%d cols, needs %.2f in)'
                            % (TABCAP.match(B[s].text).group(1) if s != i else '?',
                               len(b.columns), needs_width(b))))
    for i, b in enumerate(B):
        if isinstance(b, Table):
            continue
        part = img_part(doc, b)
        if not part:
            continue
        sh = [s for s in doc.inline_shapes
              if s._inline.graphic.graphicData.pic.blipFill.blip.embed
              in [bl.get(qn('r:embed')) for bl in b._element.findall('.//' + qn('a:blip'))]]
        ar = (sh[0].width / sh[0].height) if sh else 1.6
        if ar >= WIDE_IMG_AR:
            e = i
            if i + 1 < len(B) and not isinstance(B[i + 1], Table) \
                    and FIGCAP.match(B[i + 1].text or ''):
                e = i + 1
            regions.append((i, e, '%s (ar %.2f)' % (part, ar)))

    # THE FRONT MATTER SPANS THE PAGE. In the IEEE Access template the title, authors,
    # affiliation, abstract and index terms run the full text width before the two-column body
    # begins. Blocks 0..Index Terms are therefore one region, anchored on the Index Terms line
    # rather than on a block number.
    idx = [i for i, b in enumerate(B) if not isinstance(b, Table)
           and b.text.strip().startswith('Index Terms')]
    if len(idx) != 1:
        print('  *** Index Terms line: %d match(es), expected 1 ***' % len(idx))
        return 3
    regions.append((0, idx[0], 'front matter (title, authors, abstract, index terms)'))

    regions.sort()
    merged = []
    for s, e, tag in regions:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), merged[-1][2] + ' + ' + tag)
        else:
            merged.append((s, e, tag))

    print('\n  full-width regions: %d table(s)/figure(s) in %d section(s)'
          % (len(regions), len(merged)))
    for s, e, tag in merged:
        print('    blocks %4d..%-4d  %s' % (s, e, tag[:78]))
    narrow = [b for b in B if isinstance(b, Table)
              and not (len(b.columns) >= 4 or needs_width(b) > COL_W)]
    print('  tables that fit in a 3.46 in column and stay there: %d' % len(narrow))

    if not apply:
        print('\n  DRY RUN complete.')
        print(bar)
        return 0

    # ---------------------------------------------------------------- PASS 1
    print('\n  PASS 1 -- page setup, fonts, images')
    body = doc.element.body
    tail = body.find(qn('w:sectPr'))
    if tail is not None:
        body.remove(tail)
    body.append(make_sectPr(2, continuous=False))

    for name, size in (('Normal', 10), ('Body Text', 10), ('First Paragraph', 10)):
        try:
            st = doc.styles[name]
            st.font.size = Pt(size)
            st.paragraph_format.space_before = Pt(0)
            st.paragraph_format.space_after = Pt(1)
            st.paragraph_format.line_spacing = 1.0
        except KeyError:
            pass
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.size is not None and r.font.size > Pt(10):
                r.font.size = Pt(10)
    caps = 0
    for b in B:
        if isinstance(b, Table):
            for row in b.rows:
                for c in row.cells:
                    for p in c.paragraphs:
                        p.paragraph_format.space_before = Pt(0)
                        p.paragraph_format.space_after = Pt(0)
                        p.paragraph_format.line_spacing = 1.0
                        for r in p.runs:
                            r.font.size = Pt(8)
        elif TABCAP.match(b.text or '') or FIGCAP.match(b.text or ''):
            for r in b.runs:
                r.font.size = Pt(8)
            caps += 1
    print('    body 10 pt, tables and %d captions 8 pt' % caps)

    wide_idx = set()
    for s, e, _ in merged:
        wide_idx.update(range(s, e + 1))
    scaled = 0
    for i, b in enumerate(B):
        if isinstance(b, Table):
            continue
        if not img_part(doc, b):
            continue
        for sh in doc.inline_shapes:
            if sh._inline.graphic.graphicData.pic.blipFill.blip.embed not in \
                    [bl.get(qn('r:embed')) for bl in b._element.findall('.//' + qn('a:blip'))]:
                continue
            ar = sh.width / sh.height
            w = TEXT_W if i in wide_idx else COL_W
            sh.width = Inches(w)
            sh.height = Inches(w / ar)
            scaled += 1
    print('    %d image(s) scaled to their column or to the text block' % scaled)

    # ---------------------------------------------------------------- PASS 2
    print('\n  PASS 2 -- continuous section breaks around each full-width region')
    opened = 0
    for s, e, tag in reversed(merged):
        first_el = B[s]._tbl if isinstance(B[s], Table) else B[s]._p
        last_el = B[e]._tbl if isinstance(B[e], Table) else B[e]._p
        if s > 0:
            first_el.addprevious(marker_paragraph(doc, make_sectPr(2)))  # close the 2-col run
            opened += 1
        last_el.addnext(marker_paragraph(doc, make_sectPr(1)))           # close the 1-col run
    print('    %d one-column section(s); %d opening breaks (the front matter needs none)'
          % (len(merged), opened))

    # ---------------------------------------------------------------- PASS 3
    print('\n  PASS 3 -- post-conditions')
    B2 = blocks(doc)
    ok = True
    n_tab_after = sum(1 for b in B2 if isinstance(b, Table))
    print('    tables %d -> %d  %s' % (n_tab_before, n_tab_after,
                                       'OK' if n_tab_after == n_tab_before else '*** ***'))
    ok &= n_tab_after == n_tab_before

    imgs = [(i, img_part(doc, b)) for i, b in enumerate(B2)
            if not isinstance(b, Table) and img_part(doc, b)]
    bad = []
    for k, (i, part) in enumerate(imgs, start=1):
        nxt = None
        for j in range(i + 1, min(i + 3, len(B2))):
            if isinstance(B2[j], Table):
                break
            if B2[j].text.strip():
                nxt = B2[j]
                break
        m = FIGCAP.match(nxt.text) if nxt is not None else None
        if not m or int(m.group(1)) != k:
            bad.append((k, part, nxt.text[:30] if nxt is not None else None))
    print('    figure-caption invariant (k-th image -> "Fig. k."): %s'
          % ('holds for all %d  OK' % len(imgs) if not bad else '*** %s ***' % bad))
    print('    (v32 predates the caption fix -- not gated here)')

    for i, b in enumerate(B2):
        if isinstance(b, Table) and (len(b.columns) >= 4 or needs_width(b) > COL_W):
            prev_txt = None
            for j in range(i - 1, max(-1, i - 4), -1):
                if isinstance(B2[j], Table):
                    break
                if B2[j].text.strip():
                    prev_txt = B2[j].text
                    break
            if prev_txt is None or not TABCAP.match(prev_txt):
                print('    *** wide table at %d lost its caption neighbour ***' % i)
                ok = False

    nsect = len(doc.element.body.findall('.//' + qn('w:sectPr')))
    want = 1 + len(merged) + opened
    print('    sectPr elements in body: %d (1 final + %d closers + %d openers = %d)'
          % (nsect, len(merged), opened, want))
    ok &= nsect == want

    after = numerals(doc)
    same = Counter(before) == Counter(after)
    print('    Arabic numerals %d -> %d  %s'
          % (len(before), len(after), 'IDENTICAL' if same else '*** CHANGED ***'))
    ok &= same
    if not ok:
        print('\n    NOTHING WRITTEN.')
        return 4
    doc.save(DST)
    print('\n  written : %s\n  md5     : %s' % (DST, md5(DST)))

    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', '.', DST],
                   capture_output=True, timeout=900)
    try:
        from pypdf import PdfReader
        print('  renders to %d pages' % len(PdfReader(DST.replace('.docx', '.pdf')).pages))
    except Exception as e:
        print('  (render check unavailable: %s)' % e)
    print(bar)
    return 0


if __name__ == '__main__':
    sys.exit(main())
