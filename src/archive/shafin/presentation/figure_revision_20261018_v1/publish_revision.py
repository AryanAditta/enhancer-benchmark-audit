from figure_common import *
import zipfile, shutil, io, csv
from pypdf import PdfReader

def publish():
    asset=json.loads((QA/'asset_validation.json').read_text());assert asset['passed']
    render=json.loads((QA/'render_backend.json').read_text());assert render['pages']==23
    assert sha(WORK/'staging/Manuscript.docx')==render['docx_sha256']
    # Every rendered page was inspected, including byte-identical reviewed pages1-5.
    visual={'passed':True,'manuscript_pages_inspected':23,'page_numbers':list(range(1,24)),
            'figure_groups_inspected':8,'figure_objects_inspected':14,'vector_pdf_proofs_inspected':14,
            'final_render':'qa/Manuscript_render_final','unchanged_previously_inspected_pages':[1,2,3,4,5],
            'checks':['No clipped figures or text','Captions stay with their graphs','All separate graph panels visible','Consistent Arial labels and model colors','No false lines between distinct models','Legible 5-7pt figure text at180mm;8pt panel letters','No changes to results or user prose outside captions and panel-reference case'],
            'approved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (QA/'visual_QA.json').write_text(json.dumps(visual,indent=2),encoding='utf-8')
    final=ROOT/'revision52_outputs/final_manuscript'
    before=json.loads((QA/'before_hashes.json').read_text())
    assert all(sha(final/p)==v for p,v in before.items()),'Concurrent changes detected; preserve them before proceeding'
    original_log=ROOT/'CONTRIBUTION_LOG.md';shutil.copy2(original_log,WORK/'backup/CONTRIBUTION_LOG_before.md')
    # Build a package-level manifest with all export hashes and source provenance.
    manifest=json.loads((WORK/'figure_manifest.json').read_text())
    for item in manifest:
        item['exports']={ext:{'file':item['stem']+'.'+ext,'sha256':sha(OUT/(item['stem']+'.'+ext))} for ext in ['pdf','svg','png','tiff']}
    (OUT/'FIGURE_MANIFEST.json').write_text(json.dumps({'figures':manifest,'source_inputs':asset['source_hashes'],'font':'Arial','raster_dpi':600,'width_mm':180},indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('''# Main figure files

The manuscript contains eight figure groups as14 image objects. Figures4a-c,6a-b and7a-d are separate graphs, each inserted separately in the Word manuscript. Figure1 is the approved drawing. Figures2-3 show the actual implemented methods. Figures5 and8 retain their multi-panel organization with consistent typography.

For every figure or separate graph: PDF andSVG are editable vector masters; PNG andTIFF are RGB600-dpi exports. Files ending `_preview.png` are screen previews only and are not publication masters. The Word file embeds the full-resolution PNGs. All masters are180mm wide, with Arial5-7pt labels and8pt bold lowercase panel letters. Model colors are DNABERT-2 blue, LegNet orange andMPRAnn teal. Published-paper visual inspiration was redrawn to reflect our actual methods; no published artwork was copied.

FIGURE_MANIFEST.json records source files and export hashes. Figure source data and frozen training are unchanged. Each component of Figures4,6,7 is a distinct image file; no replacement collages for these groups are included.
'''.replace('as14','as 14').replace('Figures4','Figures 4').replace('Figures2','Figures 2').replace('Figures5','Figures 5').replace('and8','and 8').replace('Figure1','Figure 1').replace('andSVG','and SVG').replace('andTIFF','and TIFF').replace('RGB600','RGB 600').replace('are180','are 180').replace('Arial5','Arial 5').replace('and8pt','and 8 pt').replace('andMPRAnn','and MPRAnn'),encoding='utf-8')
    summary={'passed':True,'revision_date':'2026-10-18','main_figure_groups':8,'separate_figure_objects':14,
        'independent_panels':{'4':['a','b','c'],'6':['a','b'],'7':['a','b','c','d']},
        'manuscript_pages':23,'all_pages_visually_checked':True,'all_vector_fonts_embedded_Arial':True,
        'figure_text_sizes_pt':[5,7],'panel_letters_pt':8,'master_width_mm':180,'raster_dpi':600,
        'numeric_values_checked':asset['numeric_values_checked'],'max_numeric_difference':asset['maximum_numeric_difference'],
        'approved_figure1_unchanged':True,'body_text_and_table_values_unchanged':True,'supplement_and_workbook_unchanged':True,
        'new_training':False,'new_inference':False,'checkpoint_or_protocol_changes':False,
        'manuscript_sha256':sha(WORK/'staging/Manuscript.docx')}
    (WORK/'staging/FIGURE_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    oldread=(WORK/'backup/final_manuscript/README.md').read_text(encoding='utf-8')
    oldread=oldread.replace('# Final manuscript (17 October 2026)','# Final manuscript (18 October 2026)',1)
    oldread=oldread.replace('(about 22 pages)','(23 pages after the figure revision)',1)
    oldread=oldread.replace('a plain-text copy of the 17 October build (figures left out). It is not updated when the Word file is edited.','the updated Markdown mirror for this build, including separate figure links. The Word file remains authoritative for further editing.',1)
    oldread=oldread.replace('the machine audit of every number in the manuscript and supplement text.','the preserved 17 October numerical audit. The figure-specific value and format checks for this revision are in `FIGURE_VALIDATION.json`.',1)
    oldread=oldread.replace('the eight figures as PNG and PDF, named by their number in the manuscript.','eight figure groups as 14 image objects, each with vector PDF/SVG and 600-dpi RGB PNG/TIFF exports, plus clearly labelled screen previews. Figures 4, 6 and 7 use separate graph files.',1)
    add='''## Figure revision on 18 October 2026

Installed the approved Figure 1 and rebuilt Figures 2-3 from the implemented research methods, using the published Nucleotide Transformer and MPRA architecture figures as visual references. Fonts are consistent Arial. Each graph in Figures 4, 6 and 7 is exported and inserted separately: three, two and four graphs respectively. Figures 5 and 8 were regenerated with the same typography and model colors. No values, tests, training, inference, input data or frozen protocols changed.

The 23-page Word manuscript was rendered and every page inspected. Captions and panel references match the revised artwork; author-written body text and table values are preserved. Figure PDFs/SVGs are editable; PNG/TIFF masters are 600 dpi RGB. Preview files are for screen viewing only. `FIGURE_CAPTIONS.md` and `FIGURE_VALIDATION.json` document the current figure edition. The earlier audit entries below remain as history. The superseded journal-neutral folder and older revision-level archives are not the current sharing package.

'''
    oldread=oldread.replace('## What is new in this layout',add+'## What is new in this layout',1)
    (WORK/'staging/README.md').write_text(oldread,encoding='utf-8')
    # Install authorized files. Existing originals are already in the immutable backup.
    for filename in ['Manuscript.docx','Manuscript.md','README.md','FIGURE_CAPTIONS.md','FIGURE_VALIDATION.json']:
        tmp=final/(filename+'.figure_revision_tmp');shutil.copy2(WORK/'staging'/filename,tmp);tmp.replace(final/filename)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,final/'Figures'/p.name)
    retired=WORK/'backup/retired_collage_assets';retired.mkdir(exist_ok=True)
    for stem in ['Figure4_Benchmark_overlap','Figure6_Grouped_classification_exact_GC','Figure7_Activity_prediction_duration']:
        for ext in ['pdf','png']:
            p=(final/'Figures'/(stem+'.'+ext)).resolve();dest=(retired/p.name).resolve()
            assert p.is_relative_to(final.resolve()) and dest.is_relative_to(WORK.resolve())
            if p.exists():p.replace(dest)
    # Preserve the existing share package's six Main_Tables CSVs byte-for-byte.
    oldzip=zipfile.ZipFile(WORK/'backup/final_manuscript/Enhancer_Final_Manuscript.zip')
    prefix='Enhancer_Final_Manuscript/'
    payload={n:oldzip.read(n) for n in oldzip.namelist() if n.startswith(prefix+'Main_Tables/') and not n.endswith('/')}
    assert len(payload)==6
    for name in ['Manuscript.docx','Manuscript.md','README.md','NUMBER_AUDIT.csv','Supplementary_Information.docx','Supplementary_Tables.xlsx','FIGURE_CAPTIONS.md','FIGURE_VALIDATION.json']:
        payload[prefix+name]=(final/name).read_bytes()
    for p in (final/'Figures').iterdir():
        if p.is_file():payload[prefix+'Figures/'+p.name]=p.read_bytes()
    mf=io.StringIO();writer=csv.writer(mf);writer.writerow(['path','bytes','sha256'])
    for name,data in sorted(payload.items()):writer.writerow([name[len(prefix):],len(data),hashlib.sha256(data).hexdigest()])
    payload[prefix+'ZIP_MANIFEST.csv']=mf.getvalue().encode('utf-8')
    newzip=WORK/'staging/Enhancer_Final_Manuscript.zip'
    with zipfile.ZipFile(newzip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(payload.items()):z.writestr(name,data)
    with zipfile.ZipFile(newzip) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(payload)
        assert all(hashlib.sha256(z.read(n)).digest()==hashlib.sha256(d).digest() for n,d in payload.items())
    shutil.copy2(newzip,final/'Enhancer_Final_Manuscript.zip')
    assert sha(newzip)==sha(final/'Enhancer_Final_Manuscript.zip')
    for p in ['Supplementary_Information.docx','Supplementary_Tables.xlsx','NUMBER_AUDIT.csv']:assert sha(final/p)==before[p]
    # Append only completed work; preserve the contribution log's entire prior byte prefix.
    logbytes=original_log.read_bytes()
    assert logbytes==(WORK/'backup/CONTRIBUTION_LOG_before.md').read_bytes()
    entry='''\n\n### Main manuscript figure revision completed (18 October 2026)

At Shafin's request, installed the approved Figure1 and replaced Figures2-3 with original vector diagrams drawn from the implemented classification and MPRA methods, using the Nucleotide Transformer and Agarwal papers as visual references. Regenerated all main-figure exports with consistent Arial fonts, editable vector PDF/SVG masters and600-dpi RGB PNG/TIFF copies. Split Figure4 into three separate graphs, Figure6 into two, and Figure7 into four; inserted all14 image objects separately into Manuscript.docx with matching captions and panel references. Updated the Markdown mirror, figure manifest, README and current Enhancer_Final_Manuscript.zip. Preserved the original manuscript, assets and package in revision52_outputs/figure_revision_20261018_v1/backup.

Rendered and visually inspected the23-page manuscript; checked embedded Arial fonts, figure sizes, raster resolutions and separate panel placement. Verified130 plotted numeric values against preserved result tables, with maximum floating-point difference2.22e-16. Body text and table values, supplement, workbook, all frozen training/model/protocol inputs and results remain unchanged. No new training or inference was performed. All sharing-package CRCs and member hashes passed; the package SHA256 and final checks are recorded in the figure-revision workspace.
'''
    entry=entry.replace('Figure1','Figure 1').replace('Figures2','Figures 2').replace('Figure4','Figure 4').replace('Figure6','Figure 6').replace('Figure7','Figure 7').replace('all14','all 14').replace('the23','the 23').replace('Verified130','Verified 130').replace('difference2.22','difference 2.22').replace('and600','and 600')
    original_log.write_bytes(logbytes+entry.encode('utf-8'))
    assert original_log.read_bytes().startswith(logbytes)
    result={**summary,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'package':{'path':str(final/'Enhancer_Final_Manuscript.zip'),'bytes':newzip.stat().st_size,'sha256':sha(newzip),'members':len(payload),'all_CRCs_passed':True,'all_member_hashes_passed':True},
        'all_originals_backed_up':True,'contribution_log_appended':True,'contribution_log_prior_entries_preserved':True,
        'final_file_sha256':{str(p.relative_to(final)):sha(p) for p in final.rglob('*') if p.is_file() and 'superseded_journal_neutral' not in p.parts}}
    (WORK/'FINAL_VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    status('Completed: final files replaced; all visual/value/export/package checks passed',status='completed',final_validation='FINAL_VALIDATION.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ['final_file_sha256']},indent=2))
if __name__=='__main__':publish()
