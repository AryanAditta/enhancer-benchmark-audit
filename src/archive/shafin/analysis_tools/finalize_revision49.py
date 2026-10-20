"""Validate and seal the local feedback revision; never publishes or submits."""
from pathlib import Path
import datetime, hashlib, json, re, shutil, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis_tools import revise_manuscript49 as revision
revision.runtime()
from analysis_tools import final_synthesis as original
from analysis_tools.final_delivery import seal, manifest
import pandas as pd
from pypdf import PdfReader

OUT=revision.OUT
DEST=OUT/'submission_package'
PHASE=ROOT/'revision49_outputs'

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    assert Path.cwd()==ROOT
    original.verify_inputs()
    before=json.loads((PHASE/'before_revision_manifest.json').read_text(encoding='utf-8'))
    for row in before:
        assert original.sha(ROOT/row['path'])==row['sha256'],row['path']
    notebook=ROOT/'[49] Enhancer_Manuscript_Feedback_Corrections.ipynb'
    n=json.loads(notebook.read_text(encoding='utf-8'))
    codes=[c for c in n['cells'] if c['cell_type']=='code']
    assert codes and all(c.get('execution_count') is not None for c in codes)
    assert not any(o.get('output_type')=='error' for c in codes for o in c.get('outputs',[]))
    qa=json.loads((OUT/'QA/visual_QA.json').read_text(encoding='utf-8'))
    assert qa['passed'] and qa['main_text_pages_inspected']==17 and qa['supplement_pages_inspected']==15
    for row in qa['inspected_files']:
        assert original.sha(OUT/row['path'])==row['sha256'],row['path']
    reproduction=json.loads((OUT/'QA/revision_reproducibility.json').read_text(encoding='utf-8'))
    assert reproduction['passed'] and reproduction['validation']['figure_validation.json']['figures']==13
    assert all(r['pixel_identical'] for r in reproduction['validation']['figure_validation.json']['checks'])
    refs=pd.read_csv(OUT/'manuscript/references.csv').fillna('')
    assert len(refs)==44 and list(refs.reference_number)==list(range(1,45))
    assert not refs.source_id.duplicated().any()
    dois=refs.loc[refs.doi.ne(''),'doi'].str.lower()
    assert not dois.duplicated().any()
    manuscript=(OUT/'manuscript/Manuscript.md').read_text(encoding='utf-8')
    supplement=(OUT/'final_supplement/Supplementary_Information_text.md').read_text(encoding='utf-8')
    for token in ['ENCFF774DYO','ENCFF631LHA','ENCSR022GQD','ENCSR382BVV','Gosai','Claude (Anthropic)','OpenAI Codex','0.8651','0.8507','0.0144','0.0245','0.6912','0.7048']:
        assert token in manuscript,token
    for bad in ['Primary9B','training,5,611','all27 shown','fromENCSR','as72,799','andS17','Percentile95%','original1kb','Fixed0.5','below20','the new the final synthesis','sharp conditional exchangeability null']:
        assert bad not in manuscript+supplement,bad
    checked=pd.read_csv(OUT/'claim_audit/actual_manuscript_paragraph_checks.csv')
    assert len(checked)==33 and checked.numeric_evidence_reused_or_independently_verified.all()
    ledger=pd.read_csv(OUT/'claim_audit/claim_evidence_ledger.csv')
    assert (ledger.status=='VERIFIED').sum()==86
    assert len(PdfReader(OUT/'manuscript/Manuscript.pdf').pages)==23
    assert len(PdfReader(OUT/'final_supplement/Supplementary_Information.pdf').pages)==15
    for rel in ['manuscript/Manuscript.docx','final_supplement/Supplementary_Tables.xlsx']:
        with zipfile.ZipFile(OUT/rel) as archive:assert archive.testzip() is None
    assert original.sha(OUT/'final_supplement/Supplementary_Tables.xlsx')==original.sha(ROOT/'final_supplement/Supplementary_Tables.xlsx')
    assert (DEST/'Data_and_Code_Availability_DRAFT.md').read_text(encoding='utf-8').count('## Specific data sources')==1
    policy=dict(passed=True,original_artifacts_unchanged=len(before),protected_evidence_lock_passed=True,notebook49_executed_without_errors=True,verified_references=44,main_paragraphs_checked=33,verified_claim_records=86,manuscript_pages=23,supplement_pages=15,new_training=False,new_hypothesis_tests=0,full_statistics_rerun_in_revision=False,human_section_review_pending=True,author_declarations_pending=True,public_upload=False,submission=False)
    dump(OUT/'QA/revision_integrity.json',policy)
    validation=DEST/'Evidence_and_Methods/validation'
    (validation/'README.md').write_text('''# Validation scope

The revision49 directory contains checks of the current revised manuscript, supplement, figures and descriptive seed comparison. Read revision49/visual_QA.json and revision49/revision_integrity.json for this version.

The phase43, phase44, phase46 and phase48 reports are preserved historical records of the earlier draft and its layout. Their page counts and earlier figure hashes do not describe this revision. The original supplementary workbook is byte-identical; its phase44 cell-value check still applies. Phase47 records the previous full 2,000-bootstrap/9,999-permutation reproduction. Revision49 verifies the statistical code and fixed input hashes are unchanged and reruns metric/point-contrast, figure and new descriptive seed checks; it does not claim to rerun those full simulation draws.
''',encoding='utf-8')
    current=validation/'revision49';current.mkdir(parents=True,exist_ok=True)
    for file in ['visual_QA.json','revision_integrity.json','revision_reproducibility.json','supplement_summary.json']:
        shutil.copy2(OUT/'QA'/file,current/file)
    evidence=DEST/'Evidence_and_Methods'
    for file in ['single_seed_independent_reproduction.csv','revision_numeric_claims.csv']:
        shutil.copy2(OUT/'claim_audit'/file,evidence/file)
    citations=evidence/'citation_evidence';citations.mkdir(exist_ok=True)
    for file in ['data_sources.json','evaluation.json','legacy_benchmark.json','existing_reference_metadata.json']:
        shutil.copy2(revision.NOTES/file,citations/file)
    shutil.copy2(ROOT/'reports/Enhancer manuscript citation corrections.md',citations/'Citation_Review.md')
    snapshots=evidence/'revision_source_snapshots';snapshots.mkdir(exist_ok=True)
    for file in ['revise_manuscript49.py','finalize_revision49.py']:
        shutil.copy2(ROOT/'analysis_tools'/file,snapshots/file)
    for file in ['final_presentation.py','revised_word_builder.py','export_final_docx_word.ps1']:
        p=OUT/'analysis_tools'/file
        if p.exists():shutil.copy2(p,snapshots/file)
    shutil.copy2(notebook,snapshots/notebook.name)
    dump(snapshots/'SOURCE_MANIFEST.json',manifest(snapshots))
    shutil.copy2(PHASE/'revision_policy.json',evidence/'REVISION_POLICY.json')
    shutil.copy2(PHASE/'before_revision_manifest.json',evidence/'PRESERVED_ORIGINAL_ARTIFACTS.json')
    package=PHASE/'Enhancer_Manuscript_Review_Corrections_v3.zip'
    # Validate the assembled archive before adding a completed contribution entry.
    first=seal(DEST,package)
    contribution=ROOT/'CONTRIBUTION_LOG.md';prior=contribution.read_bytes()
    marker='## Revision 49 — manuscript feedback corrections (15 October 2026)'
    entry='''

## Revision 49 — manuscript feedback corrections (15 October 2026)

Completed the seven requested feedback corrections in revision49_outputs/revised, preserving all 204 snapshotted earlier manuscript/package artifacts and the protected scientific inputs. Added authenticated ENCODE HepG2 ENCFF774DYO/ENCSR022GQD and K562 ENCFF631LHA/ENCSR382BVV provenance, Agarwal et al. (2025), and the Gosai et al. (2024) Supplementary Table 12 DHS-natural source. Expanded and normalized the bibliography to 44 verified references, including 14 papers in the benchmark lineage and Whalen et al. (2022). The text distinguishes published benchmark reuse from demonstrated byte identity of the supplied local copies.

The AI disclosure names OpenAI Codex (OpenAI) and Claude (Anthropic), including the user-reported Claude notebook/handbook/erratum/results-organization assistance. Reworked the prose and supplied 19 separate sections with an explicit pending human-review worksheet; this does not claim Aditta or any author has approved or rewritten the draft.

Independently recalculated 12 individual-seed pooled AUROCs and four seed summaries from saved predictions. Stage 1 DNABERT-2 mean single-seed pooled AUROC is 0.865107 (SD 0.002119), compared with the archived single-run spaced-mer score 0.850743: gap 0.014364, versus the ensemble gap 0.024544. Stage 2 mean single-seed AUROC is 0.691167, below spaced-mer 0.704821, while the ensemble reaches 0.711711. These descriptive additions use pooled aggregation throughout and add no hypothesis tests.

Replaced Figure 1 with a connected symbolic workflow and Figure 2C with class-count bars; removed model-connecting lines from Figure 6 and added pooled individual-seed/ensemble comparisons in Figure 4. Corrected glued prose numbers and Table 1 notes. The executed notebook49 has no errors. Checked all 17 manuscript text/reference/table/legend pages, six main figures and all 15 supplementary PDF pages. The manuscript review PDF is 23 pages. The original 19-sheet supplementary workbook is unchanged because its seed table already contains the source values; the new descriptive CSV makes the aggregation explicit.

The portable revision reproduced 18 classification and 20 MPRA ensemble metric rows, 14 point contrasts, 75 weighted/literal checks, all four new seed summaries and all 13 PNG pixel arrays. Statistical implementation and fixed input hashes match the earlier fully reproduced 2,000/9,999-draw tests; those full draws were not repeated for this reporting revision. No training, checkpoint, selection or inferential-family policy changed.

Current files: revision49_outputs/revised/manuscript/; revision49_outputs/revised/final_supplement/; revision49_outputs/revised/submission_package/; [49] Enhancer_Manuscript_Feedback_Corrections.ipynb. Revised reproduction candidate: revision49_outputs/Enhancer_Reproducibility_Release_Candidate_v3.zip. Revised compact review bundle: revision49_outputs/Enhancer_Manuscript_Review_Corrections_v3.zip. CRC/member-hash checks and final SHA256 are documented in revision49_outputs/FINAL_VALIDATION.json. LATEST_MANUSCRIPT.md identifies the current copy. All earlier contribution-log entries are retained. Human section review, final author declarations, rights/public deposit and any journal submission remain pending; nothing was uploaded or submitted.
'''
    if marker not in prior.decode('utf-8-sig'):
        with contribution.open('ab') as stream:stream.write(entry.encode('utf-8'))
    assert contribution.read_bytes().startswith(prior)
    shutil.copy2(contribution,evidence/'CONTRIBUTION_LOG.md')
    final=seal(DEST,package)
    portable=json.loads((OUT/'release_package.json').read_text(encoding='utf-8'))
    assert original.sha(ROOT/portable['path'])==portable['sha256']
    with zipfile.ZipFile(ROOT/portable['path']) as archive:assert archive.testzip() is None
    for row in before:assert original.sha(ROOT/row['path'])==row['sha256'],row['path']
    pointer='''# Current manuscript review copy

The current feedback revision is **revision49_outputs/revised**. Earlier manuscript and submission-package directories remain preserved historical drafts.

- [Revised manuscript PDF](revision49_outputs/revised/manuscript/Manuscript.pdf)
- [Editable Word manuscript](revision49_outputs/revised/manuscript/Manuscript.docx)
- [Feedback response](revision49_outputs/revised/submission_package/Feedback_Response.md)
- [Section review worksheet](revision49_outputs/revised/manuscript/SECTION_REVIEW.csv)
- [Supplementary information](revision49_outputs/revised/final_supplement/Supplementary_Information.pdf)
- [Compact review bundle](revision49_outputs/Enhancer_Manuscript_Review_Corrections_v3.zip)
- [Validation and archive hashes](revision49_outputs/FINAL_VALIDATION.json)

44 references; single-seed and ensemble results separated; revised Figures 1, 2, 4 and 6; both AI tools disclosed. Journal-neutral editorial draft. Aditta's section-by-section human review and final author declarations remain pending. No upload or submission performed.
'''
    (ROOT/'LATEST_MANUSCRIPT.md').write_text(pointer,encoding='utf-8')
    result=dict(**policy,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),visual_QA_sha256=original.sha(OUT/'QA/visual_QA.json'),contribution_log_sha256=original.sha(contribution),prior_log_bytes_retained=len(prior),review_archive=final,reproduction_archive=portable,revision_source_sha256=original.sha(ROOT/'analysis_tools/revise_manuscript49.py'),notebook_sha256=original.sha(notebook))
    dump(PHASE/'FINAL_VALIDATION.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
