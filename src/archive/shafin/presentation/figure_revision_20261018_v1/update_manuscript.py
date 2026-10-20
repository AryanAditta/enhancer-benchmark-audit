"""Stage only authorized image, caption and figure-reference changes in the DOCX."""
from pathlib import Path
import copy, re, json, zipfile, hashlib, posixpath, sys, os
DEP_PY=Path(r'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python')
sys.path.append(str(DEP_PY/'Lib/site-packages'))
DLL_HANDLES=[os.add_dll_directory(str(p)) for p in [DEP_PY,DEP_PY/'DLLs'] if p.is_dir()]
from lxml import etree as E
from figure_common import WORK, ROOT, OUT, QA, status
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def q(prefix,name):return '{'+NS[prefix]+'}'+name
def txt(p):return ''.join(t.text or '' for t in p.iter(q('w','t')))
def panel_case(s):return re.sub(r'(Fig(?:ure)?\.?\s*[1-8])([A-D])',lambda m:m[1]+m[2].lower(),s)
CAPTIONS={
1:'Figure 1. Study design and scientific questions. a, Supplied legacy sequences undergo identity and similarity auditing before fixed grouped classification. b, ENCODE lentiMPRA measurements are partitioned by chromosome before the 54-fit model search. Validation selects the activity recipes, which are locked before internal evaluation and external transfer to the processed DHS-natural subset from Gosai et al. The internal branch also includes the DNABERT-2 own-prefix duration control. Arrows show data or model flow; short sequences and matrices are schematic.',
2:'Figure 2. Classification model and evaluation pipeline. a, Each 200 bp sequence uses its model-specific pretrained tokenizer, fine-tuned DNABERT or DNABERT-2 encoder, attention-mask mean pooling, dropout and a two-class linear head. b, Five fixed grouped folds and three seeds are evaluated at the fixed final second epoch. Reverse complements augment training only; scoring uses the original orientation. Dashed boxes mark descriptive single-seed and exact-GC summaries. Group-bootstrap intervals and paired permutation tests with Holm adjustment cover only headline ensemble contrasts. The seven archived baselines keep their original recipes.',
3:'Figure 3. Reporter-activity models and evaluation pipeline. a, HepG2 and K562 elements are partitioned by chromosome; targets use training-only mean and SD. b, The implemented DNABERT-2, MPRA-LegNet and MPRAnn architectures each receive three learning rates and three seeds per cell: 18 fits per model and 54 total. BN denotes batch normalization; k denotes convolutional kernel width. Schematics summarize the copied or ported implementations, not new architectures. c, Each fit retains its best finite validation-Pearson checkpoint. Mean validation Pearson over three seeds selects the learning rate; recipes and checkpoints are locked before scoring. Predictions average both orientations and then seeds. Full settings and the fixed-learning-rate own-prefix duration control are specified in Section 3.4 and Supplementary Methods.',
4:'Figure 4. Benchmark overlap and the effect of removing duplicates. a, Supplied test records with exact training identity; reverse-complement-only matches add no records. b, Sizes of retained similarity components on a logarithmic group-count axis. c, Positive and negative class counts in the supplied and duplicate-removed historical test sets. Stage 2 retains no positives after removal, so discrimination metrics are undefined. These findings concern the audited supplied copies; benchmark-lineage citations do not establish byte-identical files in every earlier study. Historical neural artifacts were unavailable, so no causal inflation estimate is made. Each panel is a separate graph.',
5:'Figure 5. Individual training seeds and ensemble comparisons. a,b, Pooled out-of-fold AUROC for each of three training seeds (circles) and the three-seed ensemble (diamonds), for Stages 1 and 2. Dotted horizontal lines show the archived single-run spaced-mer baseline on the same records. These panels use pooled AUROC throughout, not mean-fold AUROC. c, Ensemble DNABERT-2 minus DNABERT differences with unchanged 95% conditional group-bootstrap intervals. Seed variation is distinct from interval uncertainty, and the folds share training data.',
6:'Figure 6. Grouped classification and exact-GC evaluation. a, Stage 1. b, Stage 2. Each panel is a separate graph. Points show equal averages of five within-fold AUROCs from fixed predictions. Fine-tuned DNABERT rows use current three-seed ensembles; other rows are archived single runs. Error bars are descriptive fold SD, not confidence intervals or seed SD. Diamonds use all opposite-label pairs with identical GC count within each fold. Connecting segments link the two evaluations of the same model only. Matching changes eligible pairs and controls GC count only; the lower exact-GC values do not measure causal removal of all composition effects.',
7:'Figure 7. Quantitative activity prediction and duration control. a,b, HepG2 and K562 internal test results: three-seed neural ensembles (diamonds), individual neural seeds (open circles), and one archived spaced-mer baseline. c, Validation-selected LegNet minus DNABERT-2. d, DNABERT-2 longer run minus its own exact first-two-epoch prefix at learning rate 2e-5. Intervals in c,d are the unchanged conditional 95% locus-group bootstrap intervals from the final synthesis. Equal candidate counts do not establish equal compute or exhaustive tuning; grid-boundary and budget-cap caveats remain. All four panels are separate graphs.',
8:'Figure 8. External assay transfer: ranking versus absolute prediction. a, Spearman correlation. b, R². Results use selected three-seed ensembles on the DHS-natural cohort (8,399 elements per cell). Circle and square shapes distinguish HepG2 and K562; model colors are consistent across figures. Points are not connected because models are categories, not an ordered trend. Every selected model has positive rank correlation and negative R², indicating useful ranking with poor absolute prediction under assay shift. Exploratory diagnostics remain in Supplementary Fig. SF8.'}
PANEL_NAMES={
'4a':'Exact training-test identity in the supplied records.',
'4b':'Retained similarity-component sizes.',
'4c':'Historical test composition before and after duplicate removal.',
'6a':'Stage 1 grouped classification and exact-GC evaluation.',
'6b':'Stage 2 grouped classification and exact-GC evaluation.',
'7a':'HepG2 internal reporter-activity prediction.',
'7b':'K562 internal reporter-activity prediction.',
'7c':'Validation-selected model contrast.',
'7d':'Same-run DNABERT-2 duration control.'}
def ppr(p):
    a=p.find(q('w','pPr'))
    if a is None:a=E.Element(q('w','pPr'));p.insert(0,a)
    return a
def prop(parent,tag,**attrs):
    e=parent.find(q('w',tag))
    if e is None:e=E.SubElement(parent,q('w',tag))
    for k,v in attrs.items():e.set(q('w',k),str(v))
    return e
def set_caption(p,s,mini=False,last=False):
    for x in list(p):
        if x.tag!=q('w','pPr'):p.remove(x)
    pp=ppr(p);prop(pp,'spacing',before=40,after=100 if not mini else 80,line=240,lineRule='auto')
    prop(pp,'keepNext',val='1' if mini and last else '0');prop(pp,'keepLines',val='1')
    prop(pp,'ind',left=0,right=0);prop(pp,'jc',val='left')
    a=s.find('. ')+2
    for part,bold in [(s[:a],True),(s[a:],False)]:
        run=E.SubElement(p,q('w','r'));rp=E.SubElement(run,q('w','rPr'))
        prop(rp,'rFonts',ascii='Arial',hAnsi='Arial');prop(rp,'sz',val=18 if mini else 19);prop(rp,'color',val='222222')
        prop(rp,'b',val='1' if bold else '0')
        t=E.SubElement(run,q('w','t'));t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=part

def build():
    status('Staging DOCX with independent graph objects, updated captions and unchanged body results')
    src=WORK/'backup/final_manuscript/Manuscript.docx'
    archive=zipfile.ZipFile(src);doc=E.fromstring(archive.read('word/document.xml'))
    original=copy.deepcopy(doc);body=doc.find('w:body',NS)
    rel=E.fromstring(archive.read('word/_rels/document.xml.rels'));rel_ns=E.QName(rel).namespace
    rel_byid={e.get('Id'):e for e in rel};old_media=set();newmedia={};removed_ids=set()
    manifest=json.loads((WORK/'figure_manifest.json').read_text());manifest.sort(key=lambda x:(x['figure'],x.get('panel') or ''))
    # Rewrite each original figure in place, preserving surrounding manuscript content.
    for number in range(1,9):
        caption=next(p for p in body if p.tag==q('w','p') and txt(p).startswith(f'Figure {number}.'))
        imagep=caption.getprevious()
        while imagep is not None and not imagep.findall('.//a:blip',NS):
            assert not txt(imagep).strip(),'Unexpected text between figure and caption';imagep=imagep.getprevious()
        assert imagep is not None
        for b in imagep.findall('.//a:blip',NS):
            rid=b.get(q('r','embed'));removed_ids.add(rid);old_media.add(posixpath.normpath('word/'+rel_byid[rid].get('Target')))
        parts=[x for x in manifest if x['figure']==number]
        for k,item in enumerate(parts):
            newp=copy.deepcopy(imagep);pp=ppr(newp)
            prop(pp,'jc',val='center');prop(pp,'spacing',before=0,after=0,line=240,lineRule='auto');prop(pp,'keepNext',val='1')
            # 180 mm figures remain within the physical page at ~18 mm margins.
            # Only the figure paragraph extends 7.45 mm beyond the 165.1 mm body column.
            prop(pp,'ind',left=-422,right=-422)
            rid=f'rIdFigureRevision{number}{item.get("panel") or ""}'
            target='media/revised_'+item['stem']+'.png';newmedia['word/'+target]=(OUT/(item['stem']+'.png')).read_bytes()
            E.SubElement(rel,'{'+rel_ns+'}Relationship',Id=rid,Type=NS['r']+'/image',Target=target)
            for b in newp.findall('.//a:blip',NS):b.set(q('r','embed'),rid)
            cx=round(item['width_mm']*36000);cy=round(item['height_mm']*36000)
            for e in newp.findall('.//wp:extent',NS)+newp.findall('.//a:xfrm/a:ext',NS):e.set('cx',str(cx));e.set('cy',str(cy))
            for e in newp.findall('.//wp:docPr',NS):
                e.set('id',str(100+len(newmedia)));e.set('name',item['stem']);e.set('descr',f'Figure {number}{item.get("panel") or ""}. '+(PANEL_NAMES.get(str(number)+(item.get('panel') or ''),'Study methodology and evaluation.')))
            body.insert(body.index(imagep),newp)
            if len(parts)>1:
                mini=E.Element(q('w','p'));set_caption(mini,f'Figure {number}{item["panel"]}. '+PANEL_NAMES[str(number)+item['panel']],True,k==len(parts)-1)
                body.insert(body.index(imagep),mini)
        body.remove(imagep);set_caption(caption,CAPTIONS[number])
    # Keep the two detailed methodology figures together after Sections 3.3-3.5.
    # This lets the methods prose fill its page instead of leaving two large gaps.
    cap2=next(p for p in body if p.tag==q('w','p') and txt(p).startswith('Figure 2.'))
    img2=cap2.getprevious()
    cap3=next(p for p in body if p.tag==q('w','p') and txt(p).startswith('Figure 3.'))
    img3=cap3.getprevious()
    assert img2.findall('.//a:blip',NS) and img3.findall('.//a:blip',NS)
    body.remove(img2);body.remove(cap2)
    body.insert(body.index(img3),img2);body.insert(body.index(img3),cap2)
    for e in list(rel):
        if e.get('Id') in removed_ids:rel.remove(e)
    # Case-only updates to panel cross references, including references split across runs.
    for p in doc.iter(q('w','p')):
        ts=list(p.iter(q('w','t')));s=''.join(t.text or '' for t in ts);changed=panel_case(s)
        if changed!=s:
            cursor=0
            for t in ts:n=len(t.text or '');t.text=changed[cursor:cursor+n];cursor+=n
    def prose(root):return [panel_case(txt(p)) for p in root.iter(q('w','p')) if txt(p).strip() and not re.match(r'^Figure [1-8][a-d]?\.',txt(p))]
    assert prose(original)==prose(doc),'Unexpected change outside figure captions/cross references'
    settings=E.fromstring(archive.read('word/settings.xml'))
    if settings.find('w:doNotAutoCompressPictures',NS) is None:E.SubElement(settings,q('w','doNotAutoCompressPictures'))
    edits={'word/document.xml':E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True),
           'word/_rels/document.xml.rels':E.tostring(rel,xml_declaration=True,encoding='UTF-8',standalone=True),
           'word/settings.xml':E.tostring(settings,xml_declaration=True,encoding='UTF-8',standalone=True)}
    dest=WORK/'staging/Manuscript.docx'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for i in archive.infolist():
            if i.filename in old_media:continue
            out.writestr(i,edits.get(i.filename,archive.read(i.filename)))
        for name,data in newmedia.items():out.writestr(name,data)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    assert len(doc.findall('.//a:blip',NS))==14
    md=(WORK/'backup/final_manuscript/Manuscript.md').read_text(encoding='utf-8')
    md_blocks={}
    for n in range(1,9):
        parts=[x for x in manifest if x['figure']==n];images=[]
        for item in parts:
            images.append(f'![Figure {n}{item.get("panel") or ""}](Figures/{item["stem"]}.png)')
            if len(parts)>1:images.append(f'Figure {n}{item["panel"]}. '+PANEL_NAMES[str(n)+item['panel']])
        new='\n\n'.join(images)+'\n\n'+CAPTIONS[n]
        md_blocks[n]=new
        pattern=rf'^\*\*Figure {n}\..*?(?=\n\n|\Z)'
        md,count=re.subn(pattern,lambda m:new,md,count=1,flags=re.M|re.S);assert count==1,(n,count)
    md=md.replace(md_blocks[2],'',1).replace(md_blocks[3],md_blocks[2]+'\n\n'+md_blocks[3],1)
    md=panel_case(md);(WORK/'staging/Manuscript.md').write_text(md,encoding='utf-8')
    (WORK/'staging/FIGURE_CAPTIONS.md').write_text('\n\n'.join(CAPTIONS.values())+'\n',encoding='utf-8')
    result={'figure_objects':14,'figure_panels':{str(i):len([x for x in manifest if x['figure']==i]) for i in range(1,9)},'prose_and_table_text_preserved':True,'caption_and_panel_reference_changes_only':True,'old_media_parts_removed':len(old_media),'new_media_parts':len(newmedia),'source_docx_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'staged_docx_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    (QA/'docx_edit_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':build()
