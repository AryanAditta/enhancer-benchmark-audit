"""Supplement, claim audit, manuscript and local release preparation.

Uses enhancer Python and read-only bundled document libraries. Every writable
runtime/cache path is placed in the contribution workspace.
"""
from pathlib import Path
import os, sys, json, re, shutil, subprocess, hashlib, zipfile, io, html
import numpy as np
import pandas as pd
from analysis_tools import final_synthesis as s
ROOT=s.ROOT
DEPS=Path(r'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies')
NODE=DEPS/'node/bin/node.exe'
SKILLS=Path(r'C:\Users\User\.codex\plugins\cache\openai-primary-runtime')

def runtime():
    folder=ROOT/'analysis_tools/final_artifact_runtime';folder.mkdir(exist_ok=True)
    for variable in ['TEMP','TMP','TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','MPLCONFIGDIR']:
        p=folder/variable.lower();p.mkdir(exist_ok=True);os.environ[variable]=str(p)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONUTF8']='1'
    lib=DEPS/'python';os.add_dll_directory(str(lib));os.add_dll_directory(str(lib/'DLLs'))
    packages=str(lib/'Lib/site-packages')
    if packages not in sys.path:sys.path.append(packages)
    return folder

def operation_marker(kind,count,fmt,output):
    if (ROOT/output).exists():return
    skill={'xlsx':'spreadsheets','docx':'documents','pdf':'pdf'}[fmt]
    tool=SKILLS/skill/'26.1011.11113/skills'/skill/'container_tools/mark_artifact_operation_started.mjs'
    subprocess.run([str(NODE),str(tool),'--operation-kind',kind,'--expected-output-count',str(count),'--output-format',fmt],cwd=ROOT,env=os.environ.copy(),check=True)
    s.dump(output,dict(operation=kind,expected_count=count,format=fmt,tool=str(tool),succeeded=True,utc=s.NOW()))

def render_text_pdf(markdown_path,pdf_path):
    runtime()
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Preformatted
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    pdfmetrics.registerFont(TTFont('StudyArial',r'C:\Windows\Fonts\arial.ttf'))
    pdfmetrics.registerFont(TTFont('StudyArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
    styles=getSampleStyleSheet()
    for style in styles.byName.values():style.fontName='StudyArial'
    styles['Normal'].fontSize=10.5;styles['Normal'].leading=15;styles['Normal'].spaceAfter=8
    styles['Normal'].allowWidows=0;styles['Normal'].allowOrphans=0
    for name in ['Title','Heading1','Heading2','Heading3']:
        styles[name].fontName='StudyArialBold';styles[name].textColor='#111111';styles[name].keepWithNext=True
    styles.add(ParagraphStyle(name='CodeStudy',fontName='Courier',fontSize=8,leading=11,spaceAfter=8))
    story=[];code=False;block=[]
    for line in (ROOT/markdown_path).read_text(encoding='utf-8').splitlines():
        if line.startswith('```'):
            if code:story.append(Preformatted('\n'.join(block),styles['CodeStudy']));block=[]
            code=not code;continue
        if code:block.append(line);continue
        if not line.strip():continue
        level=len(line)-len(line.lstrip('#'))
        text=html.escape(line[level:].strip() if level else line)
        text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
        text=re.sub(r'`(.+?)`',r'<font name="Courier">\1</font>',text)
        story.append(Paragraph(text,styles['Title' if level==1 else 'Heading1' if level==2 else 'Heading2' if level>=3 else 'Normal']))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('StudyArial',8);canvas.drawRightString(558,28,str(doc.page));canvas.restoreState()
    (ROOT/pdf_path).parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(ROOT/pdf_path),pagesize=letter,rightMargin=54,leftMargin=54,topMargin=50,bottomMargin=45,title='Supplementary methods and result notes')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def phase44():
    runtime();s.verify_inputs();assert s.jd('phase43_outputs/summary.json')['status']=='completed'
    out=ROOT/'final_supplement';out.mkdir(exist_ok=True)
    index=s.read('final_tables/table_index.csv');tables=[]
    for r in index[index.placement.eq('supplement')].itertuples():
        d=s.read(r.file)
        # JSON preserves identifiers as text and missing values as null.
        numeric=[bool(pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])) for c in d.columns]
        name=r.table[:31]
        tables.append(dict(name=name,title=r.title,columns=list(d.columns),numeric=numeric,data=json.loads(d.to_json(orient='values',double_precision=15)),source=r.source,notes=r.notes))
    assert len(tables)==18 and len({t['name'] for t in tables})==18
    payload=dict(tables=tables)
    prior_payload=s.jd('phase44_outputs/workbook_payload.json') if (ROOT/'phase44_outputs/workbook_payload.json').exists() else None
    verified=s.jd('phase44_outputs/workbook_value_validation.json') if (ROOT/'phase44_outputs/workbook_value_validation.json').exists() else None
    can_reuse=prior_payload==payload and verified and verified['workbook_sha256']==s.sha(ROOT/'final_supplement/Supplementary_Tables.xlsx')
    s.dump('phase44_outputs/workbook_payload.json',payload)
    if not can_reuse:
        operation_marker('create',1,'xlsx','phase44_outputs/xlsx_operation_marker.json')
        subprocess.run([str(NODE),str(ROOT/'analysis_tools/build_final_supplement.mjs')],cwd=ROOT,env=os.environ.copy(),check=True)
    figures=s.read('final_figures/figure_index.csv');sf=figures[figures.placement.eq('supplement')]
    dest=out/'Supplementary_Figures';dest.mkdir(exist_ok=True)
    for r in sf.itertuples():
        for ext in ['png','svg','pdf']:
            source=ROOT/getattr(r,ext);shutil.copy2(source,dest/source.name)
    protocol=s.jd('phase30_outputs/training_protocol.json')['protocol'];mpra=s.jd('phase34_outputs/training_protocol.json')
    methods=f'''# Supplementary methods

## Evidence generations and design

This study combines a legacy binary enhancer benchmark with quantitative reporter-activity tasks. These are different outcomes and populations. Seven classification baselines retain their authenticated original single-run predictions. The current fine-tuned DNABERT and DNABERT-2 comparison comprises a separate consistent 60-fit study. The current activity comparison comprises a validation-only 54-fit search on primary9B chromosome partitions. Older filtering, subgroup and reliability analyses refer to their original fitted models. Their scope is labelled wherever used; new selected models were not retrospectively substituted.

Every final table is generated from record-level predictions or locked source tables. Source-file hashes, manifests, model generations and exclusions are listed in TableS1 and the evidence map. Earlier sealed packages, training sources, model checkpoints, protocol and selection locks remain preserved. The current manuscript uses the later erratum alongside the sealed history rather than treating an older zero-collapse summary as a complete practical diagnostic.

## FASTA parsing and sequence identity

Historical FASTA files are parsed by accumulating all nonempty sequence lines until the next header. Sequences are uppercased and validated as A/C/G/T of length200. Source file, header, supplied split and label remain attached to each record. A SHA256 of the sequence identifies exact sequence identity; the original manifest retains sequence, SHA256 and record ID. Exact matches and reverse complements are audited separately. Canonical DNA identity is the lexicographic minimum of a sequence and its reverse complement. Headers are not used as proof of distinct sequence identity or unverified genome coordinates.

The supplied Stage1 files contain 2,968 training and400 test records; Stage2 contains1,484 and200. One hundred strong test sequences are training-identical in both tasks. Removing exact duplicate test positives leaves300 Stage1 records with100 positives and200 negatives, and100 Stage2 negatives with no positives. Cleaned Stage2 discrimination is therefore undefined. The optional three-model historical experiment was stopped because the original neural historical training implementation, checkpoints and independent-test predictions cannot be authenticated. A separately labelled archived spaced-mer rescore uses the same authenticated Phase11A historical predictions for the before/after subsets; it is not a new neural replication or a causal estimate.

## Similarity groups and fixed folds

The original benchmark graph connects sequences with exact longest common subsequence length at least160 in either orientation, together with exact/reverse-complement identity. LCS means longest common subsequence; it is not an80% local sequence-identity statistic. Connected components define groups. Records in components containing a supplied historical test record are not used as grouped-CV training candidates. This leaves2,838 Stage1 and1,376 Stage2 candidates. Historical test priority accounts for the100 exact duplicate training removals; a further30 and8 related training records are excluded. Stage1 retains two mixed-label components containing six records. These remain in the primary analysis, with an evaluation-only exclusion sensitivity. Fixed StratifiedGroupKFold assignments use five folds and the original seed42. No chromosome-disjoint classification claim follows from this sequence-only graph.

```
parse and validate original FASTA records
for each candidate sequence pair:
    link if exact/RC identity or max(LCS(a,b), LCS(a,RC(b))) >= 160
groups = connected_components(links)
assign supplied historical test priority
exclude training records in historical-test-containing components
assign fixed five stratified group folds to remaining candidates
augment training sequences only after separating each held-out fold
```

## Exact GC composition sensitivity

Within each fixed held-out fold, enumerate every opposite-label pair with exactly identical GC nucleotide count in the200bp sequence. A correctly ordered score pair contributes1, a tie contributes0.5 and a reversed pair contributes0. Divide by the number of eligible pairs and average the five fold estimates equally. This is a within-GC pair AUROC, not an evaluation on a newly matched random sample. Records without an opposite-label partner at that GC count in their fold do not contribute. Matching controls GC count only; it does not control CpG content, source lineage or every composition feature. The original seven baseline values were reproduced, and the same fixed rule was applied to current fine-tuned ensemble predictions without retraining.

```
for each fixed held-out fold:
    numerator = 0; denominator = 0
    for each observed GC count:
        positive_scores = scores with label1 and this GC count
        negative_scores = scores with label0 and this GC count
        add all positive-negative pair concordances to numerator
        add number of positive-negative pairs to denominator
    fold_matched_AUC = numerator / denominator
report equal average of the five fold_matched_AUC values
```

## Classification implementations and repeated seeds

The current models use pinned pretrained revisions: DNABERT-2 {protocol['model_revisions']['dnabert2']} and DNABERT {protocol['model_revisions']['dnabert_legacy']}. The classifier uses attention-mask mean pooling including unpadded special tokens, dropout0.1 and a768-to2 linear head. Training uses cross-entropy, AdamW with learning rate{protocol['lr']}, weight decay{protocol['weight_decay']}, batch8, accumulation2 and gradient clipping1. FP16 autocast and GradScaler are used with float32 parameters. Losses are divided by2, including the terminal partial accumulation, as recorded in the frozen implementation. The fixed final second epoch is used; no outer-fold OOF outcome selects epochs, thresholds or settings. Training receives reverse-complement augmentation; scoring uses the original orientation only.

Base seeds are {', '.join(map(str,protocol['base_seeds']))}. Effective seeds follow base_seed +100*stage +fold. Within each fold the three effective seeds are distinct, but reuse across some different folds and shared training examples prevents treating all15 seed×fold points as independent replicates. TableS2 lists the actual schedule and metrics. The primary classification threshold remains0.5. AUROC, average precision, Brier score, sensitivity, specificity, MCC and balanced accuracy are recomputed from fixed predictions. Mean-fold AUROC, pooled OOF AUROC, fold SD, seed SD and three-seed ensemble performance are explicitly distinguished.

GC/CpG logistic regression uses GC fraction and CpG frequency. The3-mer and spaced6/4-mer models use their locked original feature and preprocessing recipes. The spaced representation enumerates all15 four-position masks in each six-base window, with3,840 features; IDF and other fitted preprocessing remain training-only. Small-CNN and frozen language-model baseline predictions remain authenticated archived single runs. Their complete original configuration and provenance files accompany the reproducibility inputs; no repeated-seed uncertainty is manufactured.

## MPRA construction and population exclusions

The primary tasks retain paired-orientation200bp reporter elements with complete activity measurements under the original construction protocol. Row-level source/design, chromosome, length and missing-orientation exclusions are distinguished from candidate-element exclusions in TableS10. The complete-pair pool contains149,632 candidates across the two cells;76,833 are retained and72,799 are removed by the original legacy overlap screen. The screen is largely shared15-mers and must not be described as72,799 full-sequence duplicates. Filtering changes the evaluation population. The removed-candidate transport analyses score frozen test-chromosome candidates with older fitted models and do not estimate a causal effect of retraining without the filter.

Primary9B chromosome membership, IDs and locus groups are frozen. HepG2 has18,665 training,5,611 validation and9,048 test elements; K562 has23,560,6,804 and13,145. Training-only target mean and population SD are calculated in float64, and normalized targets are stored as float32 for fitting. Each prediction is converted back to activity units using its own train-only scaling. All model fits use the same declared partitions for a cell. Existing cohorts were examined before this follow-up and are not described as newly independent confirmatory tests.

## Activity architectures and fitting recipes

DNABERT-2 uses the pinned pretrained encoder, an attention-mask pooled representation, dropout0.1 and a scalar linear regressor. Tokenization uses the actual BPE tokenizer without truncation of these200bp sequences and verifies at most256 tokens. The copied human MPRA LegNet implementation uses four DNA input channels, a64-channel stem with kernel11, efficient residual-concatenation blocks with widths80/96/112/128, kernel9, four pooling factors2 and resize factor4; the mapper and head are retained exactly from the copied architecture. MPRAnn is the copied PyTorch port of the original standard CNN: four valid convolutions with250/250/250/100 channels and kernels7/8/3/2, activation/BatchNorm order and channel softmax as specified in source, one max-pool factor2, dense300/200 layers and a scalar output. Keras-equivalent BN epsilon/momentum and initialization are documented in that source. Neither specialist architecture was redesigned for these test results.

Declared learning-rate grids are DNABERT-2 [1e-5,2e-5,5e-5], LegNet maximum rates [0.005,0.01,0.02] and MPRAnn [0.0003,0.001,0.003]. Each recipe uses seeds {', '.join(map(str,mpra['seeds']))}. DNABERT-2 uses AdamW, batch8/accumulation2, a maximum30 epochs, patience5 and minimum8. LegNet uses AdamW with weight decay0.1, batch1,024 and a20-epoch OneCycle schedule with pct_start0.3 and initial rate max_lr/25. MPRAnn uses Adam epsilon1e-7, batch128, maximum100 epochs and patience10. The checkpoint is the highest finite validation Pearson, with earliest epoch resolving ties; early-stopping patience requires improvement above0.0001. Checkpoint selection and patience are distinct rules. Detailed update counts, skipped AMP steps, epochs, memory and timings remain visible in TablesS7 andS17.

Training uses deterministic per-epoch shuffling and the original model-specific reverse-complement policies. DNABERT-2 keeps the partial terminal batch; CNNs use floor batches as in the inherited recipe. Inference averages forward and reverse-complement predictions before the three-seed ensemble average. Every DNABERT-2 fit also retains its validation-best checkpoint within its first two epochs. The primary duration control compares each fixed-LR2e-5 full trajectory to that exact new-run prefix; it is not a comparison with an older historical run.

## Validation-only search and practical failure accounting

For each model/cell, choose the learning rate with greatest mean validation Pearson over all three declared seeds. Any undefined seed makes that candidate ineligible; lowest grid index resolves deterministic ties. All54 fits finish before checkpoint/recipe hashes are locked and new test or external inference starts. There is no test-driven change to grids, stopping, augmentation, checkpoint rules, exclusions or thresholds. Equal candidate/seed counts do not imply equal compute, pretraining, epoch count, updates or exhaustive tuning.

The frozen flag has zero undefined-Pearson fits. A later descriptive SD<1e-4 screen identifies two practical near-constant DNABERT-2 LR5e-5 runs, one per cell, seed20260932. Their Pearson scores were finite under the original rule. They remain in all candidate means and selection; no candidate was retroactively disqualified. Grid-boundary choices and six potentially budget-limited selected fits are disclosed, not used to claim convergence. The earlier historical9C collapse remains a different model/run with unresolved root cause and is not repaired by this9B study.

## External transfer and failure diagnostics

The current selected-model comparison uses the frozen8,399 DHS-natural sequences per cell. The original external construction and origin inventory are retained. Canonical-DNA separation, specified k-mer screens and natural-coordinate checks address their stated forms of overlap; they do not prove absence of every homology or pretraining membership. Spearman is the external headline, while Pearson, MAE, RMSE, R2 and prediction-minus-measurement mean error distinguish ranking from absolute-scale calibration. Other external origin scores in TableS9 are explicitly older fitted-model diagnostics.

Older retained/removed subgroup analyses use training-only full-precision thresholds, with integer CG-count definitions to avoid boundary ambiguity from rounded exports. Small-group suppression and all denominators remain visible. The prediction-interval study uses its frozen disjoint calibration/audit design with constant-width and seed-scaled intervals. Nominal marginal coverage does not ensure conditional coverage for every subgroup or individual. Activity-tail diagnostics use measured target bands and therefore describe failure modes rather than a prospective target-free selection procedure.

## Final statistical synthesis

Fourteen contrasts were fixed for final synthesis after earlier outcomes had already been seen. They comprise DNABERT-2 minus DNABERT and minus spaced-mer for each classification task; LegNet minus DNABERT-2 and DNABERT-2 minus MPRAnn for each cell internally and externally; and two internal fixed-LR DNABERT-2 duration controls. This is not prospective registration of the entire research project.

Percentile95% conditional intervals use2,000 paired whole-group bootstrap draws. Internal MPRA resamples frozen locus groups; classification resamples locked similarity-graph components; external transfer resamples canonical-DNA groups. Spearman ranks are recomputed over duplicated resampled observations, including ties. A weighted implementation was checked against literal duplicated-row AUROC, Pearson and Spearman calculations. The original1kb external-region sensitivity remains separately labelled in the archived paired interval table.

P-values use9,999 whole-group paired prediction swaps and the plus-one Monte Carlo correction, with a two-sided absolute-difference statistic. The null is sharp conditional exchangeability of the two fitted prediction vectors within groups, not equality or equivalence of learning algorithms. Holm adjustment is applied across the14-contrast family. Monte Carlo resolution and standard errors, defined/undefined draw counts and all contrast signs are retained. Pointwise bootstrap intervals and Holm-adjusted p-values answer related but different questions. Other historical secondary intervals remain labelled exploratory and unadjusted; no p-value is reverse-engineered from an interval. All inference conditions on fitted models and previously examined cohorts and excludes model-search, training-algorithm and pretraining uncertainty.
'''
    for a,b in {'and400':'and 400','contains1,484':'contains 1,484','and200':'and 200','leaves300':'leaves 300','with100':'with 100','and100':'and 100','100Stage2':'100 Stage2','2,838Stage1':'2,838 Stage1','and1,376':'and 1,376','further30':'further 30','and8 ':'and 8 ','seed42':'seed 42','dropout0.1':'dropout 0.1','clipping1':'clipping 1','by2,':'by 2,','contains149,632':'contains 149,632','candidates;76,833':'candidates; 76,833','and72,799':'and 72,799','largely shared15-mers':'largely shared 15-mers','batch8/':'batch 8/','maximum30':'maximum 30','patience5':'patience 5','minimum8':'minimum 8','batch1,024':'batch 1,024','maximum100':'maximum 100','patience10':'patience 10','above0.0001':'above 0.0001','All54':'All 54','frozen8,399':'frozen 8,399','use2,000':'use 2,000','use9,999':'use 9,999','the14-contrast':'the 14-contrast'}.items():methods=methods.replace(a,b)
    methods=re.sub(r'\b(length|least|an|leaves|the|contributes|a|rate|decay|batch|accumulation|all|remains|The|spaced|enumerates|with|orientation|cells|has|and|these|most|kernel|widths|factors|factor|dense|pct_start|epsilon|seed|LR|primary|historical)(?=\d)',r'\1 ',methods)
    methods=re.sub(r'\b(Table|Tables)(?=S\d)',r'\1 ',methods)
    methods=methods.replace('Stage1','Stage 1').replace('Stage2','Stage 2').replace('768-to2','768-to-2')
    methods=re.sub(r'(\d)bp\b',r'\1 bp',methods)
    s.text('final_supplement/Supplementary_Methods.md',methods)
    notes='# Supplementary result notes\n\n'+'\n\n'.join('## Table '+r.table.split('_',1)[0]+' '+r.title+'\n\n'+r.notes for r in index[index.placement.eq('supplement')].itertuples())
    notes+='\n\n## Supplementary figure captions\n\n'+'\n\n'.join('### Figure '+r.figure.split('_',1)[0]+' '+r.title+'\n\n'+r.caption for r in sf.itertuples())
    s.text('final_supplement/Supplementary_Result_Notes.md',notes)
    combined='final_supplement/Supplementary_Information_text.md';s.text(combined,methods+'\n\n'+notes)
    operation_marker('create',2,'pdf','phase44_outputs/pdf_operation_marker.json')
    render_text_pdf(combined,'phase44_outputs/Supplementary_Information_text.pdf')
    from pypdf import PdfReader,PdfWriter
    writer=PdfWriter();writer.append(str(ROOT/'phase44_outputs/Supplementary_Information_text.pdf'))
    for r in sf.itertuples():writer.append(str(ROOT/r.pdf))
    writer.add_metadata({'/Title':'Supplementary information for enhancer benchmark and reporter activity evaluation'})
    with (out/'Supplementary_Information.pdf').open('wb') as f:writer.write(f)
    reader=PdfReader(str(out/'Supplementary_Information.pdf'))
    assert len(reader.pages)>len(sf) and all(page.extract_text() for page in reader.pages)
    s.dump('phase44_outputs/summary.json',dict(status='built_pending_workbook_and_PDF_visual_QA',supplementary_tables=18,supplementary_figures=7,pdf_pages=len(reader.pages),workbook=str((out/'Supplementary_Tables.xlsx').relative_to(ROOT)),source_sha256=s.sha(__file__)))
    print('Phase44 supplement built; workbook and PDF visual QA pending.')
