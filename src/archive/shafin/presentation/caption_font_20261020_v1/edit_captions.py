from pathlib import Path
from zipfile import ZipFile
import sys, json, hashlib, shutil, re, copy

ROOT=Path(r"D:\Aditta Thesis\For Shafin\Shafin's Contribution")
WORK=Path(__file__).resolve().parent
DEPS=Path(r'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies')
sys.path.append(str(DEPS/'python/Lib/site-packages'))
from lxml import etree as E
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}
q=lambda s:'{'+W+'}'+s
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'revision52_outputs/final_manuscript/Manuscript.docx'
backup=WORK/'backup/Manuscript_before_caption_font.docx'
staged=WORK/'staging/Manuscript.docx'
for path in [backup.parent,staged.parent,WORK/'qa']:path.mkdir(parents=True,exist_ok=True)
assert not backup.exists()
before_hash=sha(source)
shutil.copy2(source,backup)
assert sha(backup)==before_hash
with ZipFile(backup) as z:
    root=E.fromstring(z.read('word/document.xml'))
    body=root.find(q('body'))
    before=copy.deepcopy(root)
    captions=[]
    for index,para in enumerate(body):
        if para.tag!=q('p'):continue
        text=''.join(para.xpath('.//w:t/text()',namespaces=NS))
        match=re.match(r'^Figure ([1-8][a-d]?)\.',text)
        if not match:continue
        captions.append({'body_index':index,'figure':match[1],'text':text})
        for run in para.findall('.//'+q('r')):
            pr=run.find(q('rPr'))
            if pr is None:pr=E.Element(q('rPr'));run.insert(0,pr)
            fonts=pr.find(q('rFonts'))
            if fonts is None:fonts=E.Element(q('rFonts'));pr.insert(0,fonts)
            for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme']:
                fonts.attrib.pop(q(key),None)
            for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(q(key),'Times New Roman')
    assert len(captions)==17, 'Expected eight main legends plus nine separate panel captions.'
    assert root.xpath('//w:t/text()',namespaces=NS)==before.xpath('//w:t/text()',namespaces=NS)
    # Removing font attributes from both copies must leave identical full XML trees.
    normalized=copy.deepcopy(root)
    for tree in [before,normalized]:
        b=tree.find(q('body'))
        for item in captions:
            for fonts in b[item['body_index']].findall('.//'+q('rFonts')):
                fonts.getparent().remove(fonts)
    assert E.tostring(before,method='c14n')==E.tostring(normalized,method='c14n'), 'Only caption font properties may change.'
    for item in captions:
        for font in body[item['body_index']].findall('.//'+q('rFonts')):
            assert all(font.get(q(key))=='Times New Roman' for key in ['ascii','hAnsi','eastAsia','cs'])
    xml=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
    with ZipFile(staged,'w') as out:
        for item in z.infolist():out.writestr(item,xml if item.filename=='word/document.xml' else z.read(item.filename))
with ZipFile(backup) as a,ZipFile(staged) as b:
    assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['word/document.xml']
    assert b.testzip() is None
report={'source_sha256':before_hash,'staged_sha256':sha(staged),'caption_paragraphs':len(captions),'captions':captions,'font':'Times New Roman','font_changes_only':True,'all_text_unchanged':True,'bold_sizes_alignment_and_spacing_unchanged':True,'all_figures_tables_other_content_unchanged':True,'backup':str(backup),'status':'staged_for_visual_review'}
(WORK/'qa/edit_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='captions'},indent=2))
