from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as E
import io, csv, json, hashlib, shutil, re, posixpath, os

ROOT=Path(r"D:\Aditta Thesis\For Shafin\Shafin's Contribution")
WORK=Path(__file__).resolve().parent
FINAL=ROOT/'revision52_outputs/final_manuscript'
BACKUP=WORK/'backup'
STAGE=WORK/'staging'
ARCHIVE=FINAL/'Enhancer_Final_Manuscript.zip'
PREFIX='Enhancer_Final_Manuscript/'
sha=lambda b:hashlib.sha256(b).hexdigest()
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
for directory in [BACKUP,STAGE]:directory.mkdir(parents=True,exist_ok=True)
assert not (BACKUP/ARCHIVE.name).exists(), 'Preserve existing backup; create a versioned follow-up instead.'
assert shutil.disk_usage(FINAL).free>150*1024**2
tracked=[ARCHIVE,FINAL/'README.md',FINAL/'Manuscript.md',FINAL/'Manuscript.docx',FINAL/'Supplementary_Information.docx',FINAL/'Supplementary_Tables.xlsx']
before_hashes={p.name:sha(p.read_bytes()) for p in tracked}
for p in tracked[:3]:shutil.copy2(p,BACKUP/p.name)
with ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    old={n[len(PREFIX):]:z.read(n) for n in z.namelist() if n.startswith(PREFIX) and not n.endswith('/')}
    old_main={n:b for n,b in old.items() if n.startswith('Main_Tables/')}
assert len(old_main)==6

# Keep current Word files authoritative. This task does not edit either DOCX.
new_files=dict(old_main)
for p in FINAL.iterdir():
    if p.is_file() and p!=ARCHIVE:new_files[p.name]=p.read_bytes()
for p in (FINAL/'Figures').rglob('*'):
    if p.is_file():new_files[p.relative_to(FINAL).as_posix()]=p.read_bytes()

# Refresh the Markdown mirror only for the four panel labels already changed in Word.
md=(FINAL/'Manuscript.md').read_text(encoding='utf-8')
for old_heading,new_heading in [
    ('*Legacy classification tasks*','**A. Legacy classification tasks**'),
    ('*Reporter activity populations*','**B. Reporter activity populations**'),
    ('*Supplied files*','**A. Supplied files**'),
    ('*Retained candidates after historical-test priority and related-group exclusions*','**B. Retained candidates after historical-test priority and related-group exclusions**')]:
    assert md.count(old_heading)==1
    md=md.replace(old_heading,new_heading,1)
new_files['Manuscript.md']=md.encode('utf-8')

# Add the literature comparison, so all seven main-text tables have separate CSV material.
with ZipFile(FINAL/'Manuscript.docx') as z:
    body=E.fromstring(z.read('word/document.xml')).find('w:body',ns)
    table=body.find('w:tbl',ns)
    rows=[[''.join(c.itertext()) if False else ''.join(t.text or '' for t in c.findall('.//w:t',ns)) for c in row.findall('w:tc',ns)] for row in table.findall('w:tr',ns)]
assert rows[0]==['Method (Year)','Architecture','Stage 1 MCC','Stage 2 MCC','Stage 1 AUC']
stream=io.StringIO(newline='')
writer=csv.writer(stream,lineterminator='\n');writer.writerows(rows)
new_files['Main_Tables/Table1_Published_methods.csv']=stream.getvalue().encode('utf-8-sig')
new_files['Main_Tables/README.md']=b'''# Main manuscript tables

The seven main tables are editable in Manuscript.docx. Separate CSV material:

- Table 1: Table1_Published_methods.csv (the exact displayed comparison; full literature table is supplementary S19).
- Table 2: T1_Datasets.csv.
- Table 3: T2_Benchmark_audit.csv.
- Table 4: T3_Classification.csv.
- Table 5: T4_MPRA_internal.csv.
- Table 6: T6_Headline_contrasts.csv.
- Table 7: T5_External_transfer.csv.

The six original result CSVs are preserved byte-for-byte. Their filenames predate the current main-text numbering. Supplementary Tables S1-S19 are in Supplementary_Tables.xlsx.
'''

# Extract full-resolution supplementary images without altering their bytes.
supp_manifest=[]
captions=[]
with ZipFile(FINAL/'Supplementary_Information.docx') as z:
    rels=E.fromstring(z.read('word/_rels/document.xml.rels'))
    mapping={r.get('Id'):r.get('Target') for r in rels}
    body=E.fromstring(z.read('word/document.xml')).find('w:body',ns)
    pending=[]
    for para in body:
        pending.extend(b.get('{'+ns['r']+'}embed') for b in para.findall('.//a:blip',ns))
        text=''.join(t.text or '' for t in para.findall('.//w:t',ns))
        match=re.match(r'^Supplementary Figure (SF[1-8])\.',text)
        if not match:continue
        assert len(pending)==1,(match[1],pending)
        member=posixpath.normpath(posixpath.join('word',mapping[pending[0]]))
        assert member.startswith('word/media/')
        data=z.read(member)
        destination=f'Supplementary_Figures/{match[1]}{Path(member).suffix}'
        assert destination not in new_files
        new_files[destination]=data
        captions.append(text)
        supp_manifest.append({'figure':match[1],'file':destination,'bytes':len(data),'sha256':sha(data),'source':'Supplementary_Information.docx','source_member':member,'extracted_without_image_changes':True})
        pending=[]
assert [x['figure'] for x in supp_manifest]==[f'SF{i}' for i in range(1,9)]
new_files['Supplementary_Figures/CAPTIONS.md']=('# Supplementary figure captions\n\n'+'\n\n'.join(captions)+'\n').encode('utf-8')
new_files['Supplementary_Figures/MANIFEST.json']=json.dumps(supp_manifest,indent=2).encode('utf-8')
new_files['Supplementary_Figures/README.md']=b'''# Supplementary figure files

SF1-SF8 are exact PNG image files extracted from the current Supplementary_Information.docx. No graph, value, resolution or image content was regenerated or changed. The embedded originals remain in the supplementary document. CAPTIONS.md contains the corresponding unchanged legends and MANIFEST.json records the source media members and SHA256 hashes.

These are the available embedded PNG originals; this folder does not claim new vector masters or new resolution. Main figure masters are in ../Figures/.
'''

readme=(FINAL/'README.md').read_text(encoding='utf-8')
assert readme.startswith('# Final manuscript (18 October 2026)')
readme=readme.replace('# Final manuscript (18 October 2026)','# Final manuscript sharing package (20 October 2026)',1)
insert='''
## Current sharing package checked on 20 October 2026

Send `Enhancer_Final_Manuscript.zip` for manuscript review. It contains the current 23-page Word manuscript, all eight main figure groups as 14 individual image objects in PDF/SVG/600-dpi PNG/TIFF plus previews, all seven main tables (editable in Word with separate CSV material), the supplementary Word document, all 19 supplementary tables in the workbook, and all eight supplementary figures as separate original PNG files with captions.

The current Word manuscript includes the 20 October formatting edits: bold A/B sections and horizontal rules for Tables 2 and 3, and Times New Roman for all 17 main-figure and panel captions. Only the four matching panel labels were updated in the Markdown mirror; Word remains authoritative. No research data, statistics, figure values or figure image content changed.

`FIGURE_VALIDATION.json` records the historical 18 October figure-build checks and manuscript hash; it is retained with that original date and hash. `NUMBER_AUDIT.csv` is the preserved 17 October numerical audit. `ZIP_MANIFEST.csv` records the exact current package member sizes and SHA256 hashes.

This is the compact manuscript-review package. It does not include the full research inputs, model checkpoints, training notebooks, research code archive, older manuscript layouts, backup copies or generated QA page images. Those remain preserved in the contribution workspace. The `Still open` section below records author/release/submission items; this package is for review, not a claim that submission requirements are complete.

The supplementary PNG files were extracted without modification from the current supplementary document. The six original result CSVs are preserved, and the displayed Table 1 literature comparison was added directly from the current Word table.
'''
readme=readme.replace('\n## Figure revision on 18 October 2026\n',insert+'\n## Figure revision on 18 October 2026\n',1)
new_files['README.md']=readme.encode('utf-8')

# Write the new standalone sharing assets to staging before touching the live package.
for name,data in new_files.items():
    target=STAGE/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
manifest_buffer=io.StringIO(newline='')
writer=csv.writer(manifest_buffer,lineterminator='\n');writer.writerow(['path','bytes','sha256'])
for name,data in sorted(new_files.items()):writer.writerow([name,len(data),sha(data)])
new_files['ZIP_MANIFEST.csv']=manifest_buffer.getvalue().encode('utf-8')
archive=WORK/ARCHIVE.name
with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for name,data in sorted(new_files.items()):z.writestr(PREFIX+name,data)
with ZipFile(archive) as z:
    assert z.testzip() is None
    for name,data in new_files.items():assert z.read(PREFIX+name)==data
    rows=list(csv.DictReader(io.StringIO(z.read(PREFIX+'ZIP_MANIFEST.csv').decode('utf-8'))))
    assert len(rows)==len(new_files)-1
    for row in rows:
        data=z.read(PREFIX+row['path'])
        assert len(data)==int(row['bytes']) and sha(data)==row['sha256']
    for name,data in old_main.items():assert z.read(PREFIX+name)==data
    assert z.read(PREFIX+'Manuscript.docx')==(FINAL/'Manuscript.docx').read_bytes()

# Detect concurrent edits and preserve every older version before publication.
for p in tracked:assert sha(p.read_bytes())==before_hashes[p.name],('Concurrent change',p)
new_names=[n for n in new_files if n.startswith(('Main_Tables/','Supplementary_Figures/'))]
for name in new_names:assert not (FINAL/name).exists(), ('Do not overwrite an existing uninspected asset',name)
for name in ['README.md','Manuscript.md']+new_names:
    target=FINAL/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(STAGE/name,target)
pending=FINAL/'Enhancer_Final_Manuscript.refresh.tmp'
assert not pending.exists()
shutil.copyfile(archive,pending)
assert sha(pending.read_bytes())==sha(archive.read_bytes())
os.replace(pending,ARCHIVE)
with ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    for name,data in new_files.items():assert z.read(PREFIX+name)==data
    for p in FINAL.rglob('*'):
        if not p.is_file() or p==ARCHIVE or 'superseded_journal_neutral' in p.relative_to(FINAL).parts:continue
        name=p.relative_to(FINAL).as_posix()
        assert name in new_files and z.read(PREFIX+name)==p.read_bytes(),('Missing or stale live package member',name)
for name in ['Manuscript.docx','Supplementary_Information.docx','Supplementary_Tables.xlsx']:
    assert sha((FINAL/name).read_bytes())==before_hashes[name]
report={'passed':True,'date_local':'2026-10-20','archive':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'sha256':sha(ARCHIVE.read_bytes()),'members':len(new_files),'all_CRCs_passed':True,'all_manifest_hashes_passed':True,'current_folder_files_match_archive':True,'excluded_folder':'superseded_journal_neutral','main_figure_groups':8,'main_figure_objects':14,'main_tables':7,'supplementary_figures_separate_original_pngs':8,'supplementary_tables':19,'current_manuscript_sha256':before_hashes['Manuscript.docx'],'original_archive_backup':str(BACKUP/ARCHIVE.name),'no_research_or_document_content_changed':True,'markdown_changes_only_four_existing_panel_labels':True}
(WORK/'FINAL_VALIDATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
