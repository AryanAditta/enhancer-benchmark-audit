from pathlib import Path
from zipfile import ZipFile
import sys, os, json, hashlib, shutil, copy, datetime

ROOT = Path(r"D:\Aditta Thesis\For Shafin\Shafin's Contribution")
WORK = Path(__file__).resolve().parent
DEPS = Path(r'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies')
sys.path.append(str(DEPS / 'python/Lib/site-packages'))
from lxml import etree as E

SOURCE = ROOT / 'revision52_outputs/final_manuscript/Manuscript.docx'
BACKUP = WORK / 'backup/Manuscript_before_table2_formatting.docx'
OUT = WORK / 'staging/Manuscript.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}

def q(s): return '{' + W + '}' + s
def text(el): return ''.join(el.itertext()) if False else ''.join(el.xpath('.//w:t/text()', namespaces=NS))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def child(el, tag):
    node = el.find(q(tag))
    if node is None:
        node = E.SubElement(el, q(tag))
    return node
def attr(el, **attrs):
    for name, value in attrs.items(): el.set(q(name), str(value))
def order(el, tags):
    rank = {q(tag): i for i, tag in enumerate(tags)}
    el[:] = sorted(el, key=lambda node: rank.get(node.tag, 1000))

for d in [BACKUP.parent, OUT.parent, WORK / 'qa']:
    d.mkdir(parents=True, exist_ok=True)
assert not BACKUP.exists(), 'Preserve the earlier backup; use a versioned follow-up for another edit.'
original_hash = sha(SOURCE)
shutil.copy2(SOURCE, BACKUP)
assert sha(BACKUP) == original_hash
with ZipFile(BACKUP) as z:
    raw = z.read('word/document.xml')
    root = E.fromstring(raw)
    body = root.find(q('body'))
    headings = [p for p in body.findall(q('p')) if text(p) in ['Legacy classification tasks', 'Reporter activity populations']]
    assert len(headings) == 2
    targets = []
    for p in headings:
        table = p.getnext()
        assert table.tag == q('tbl')
        targets.extend([p, table])
    indices = [body.index(el) for el in targets]
    before_children = [E.tostring(el, method='c14n') for el in body]
    before_rows = [[text(c) for c in row.findall(q('tc'))] for table in targets[1::2] for row in table.findall(q('tr'))]
    before_text = root.xpath('//w:t/text()', namespaces=NS)

    for index, heading in enumerate(headings):
        ppr = child(heading, 'pPr')
        attr(child(ppr, 'keepNext'), val='1')
        attr(child(ppr, 'keepLines'), val='1')
        attr(child(ppr, 'spacing'), before=80 if index == 0 else 160, after=60, line=240, lineRule='auto')
        attr(child(ppr, 'jc'), val='left')
        runs = heading.findall(q('r'))
        assert len(runs) == 1 and len(runs[0].findall(q('t'))) == 1
        run = runs[0]
        run.find(q('t')).text = ('A. ' if index == 0 else 'B. ') + text(heading)
        rpr = child(run, 'rPr')
        attr(child(rpr, 'b'), val='1')
        attr(child(rpr, 'bCs'), val='1')
        attr(child(rpr, 'i'), val='0')
        attr(child(rpr, 'iCs'), val='0')
        attr(child(rpr, 'color'), val='000000')
        attr(child(rpr, 'sz'), val=18)
        attr(child(rpr, 'szCs'), val=18)
        order(ppr, ['pStyle','keepNext','keepLines','pageBreakBefore','spacing','ind','jc','rPr'])
        order(rpr, ['rStyle','rFonts','b','bCs','i','iCs','color','sz','szCs'])

    for table in targets[1::2]:
        pr = child(table, 'tblPr')
        borders = child(pr, 'tblBorders')
        for old in list(borders): borders.remove(old)
        for side in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            border = E.SubElement(borders, q(side))
            attr(border, val='single' if side in ['top','bottom'] else 'nil', color='000000', sz=6 if side == 'top' else 4, space=0)
        rows = table.findall(q('tr'))
        for row_index, row in enumerate(rows):
            for column_index, cell in enumerate(row.findall(q('tc'))):
                cpr = child(cell, 'tcPr')
                cb = child(cpr, 'tcBorders')
                for old in list(cb): cb.remove(old)
                for side in ['top','left','bottom','right']:
                    border = E.SubElement(cb, q(side))
                    visible = (side == 'top' and row_index == 0) or (side == 'bottom' and row_index in [0, len(rows)-1])
                    attr(border, val='single' if visible else 'nil', color='000000', sz=6 if side == 'top' else 4, space=0)
                # Preserve the manuscript's typography, widths, cell padding and all text.
                for p in cell.findall(q('p')):
                    ppr = child(p,'pPr')
                    attr(child(ppr,'jc'), val='left' if column_index == 0 else ('center' if row_index == 0 else 'right'))

    after_rows = [[text(c) for c in row.findall(q('tc'))] for table in targets[1::2] for row in table.findall(q('tr'))]
    assert after_rows == before_rows, 'Table data must stay exactly unchanged.'
    after_text = root.xpath('//w:t/text()', namespaces=NS)
    expected_text = [('A. ' + t if t == 'Legacy classification tasks' else 'B. ' + t if t == 'Reporter activity populations' else t) for t in before_text]
    assert after_text == expected_text
    for i, element in enumerate(body):
        if i not in indices:
            assert E.tostring(element, method='c14n') == before_children[i], ('Unexpected edit outside Table 2', i)
    xml = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    with ZipFile(OUT,'w') as out:
        for entry in z.infolist(): out.writestr(entry, xml if entry.filename == 'word/document.xml' else z.read(entry.filename))
with ZipFile(BACKUP) as z, ZipFile(OUT) as out:
    changed_members = [n for n in z.namelist() if z.read(n) != out.read(n)]
    assert changed_members == ['word/document.xml']
    assert out.testzip() is None

report = {
    'date_local':'2026-10-20',
    'source_sha256':original_hash,
    'staged_sha256':sha(OUT),
    'backup':str(BACKUP),
    'changed_package_members':changed_members,
    'changed_body_element_indices':indices,
    'table_data_exactly_unchanged':True,
    'outside_table2_structurally_unchanged':True,
    'text_changes_only_two_panel_prefixes':True,
    'panels':['A. Legacy classification tasks','B. Reporter activity populations'],
    'style':'Bold upright panel headings; increased inter-panel spacing; horizontal rules above/below headings and table foot; no vertical or body gridlines; existing typeface and sizes retained.',
    'journal_style_basis':[
        'https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/',
        'https://www.nature.com/srep/author-instructions/submission-guidelines'
    ],
    'status':'staged_for_visual_review'
}
(WORK/'qa/edit_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
