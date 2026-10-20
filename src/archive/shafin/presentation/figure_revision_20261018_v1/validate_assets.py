from figure_common import *
from pypdf import PdfReader
import zipfile
def validate():
    m=json.loads((WORK/'figure_manifest.json').read_text());assert len(m)==14
    records=[]
    for item in m:
        stem=item['stem'];p=PdfReader(OUT/(stem+'.pdf'));assert len(p.pages)==1;page=p.pages[0]
        width=float(page.mediabox.width)*25.4/72;assert abs(width-180)<.01
        sizes=[]
        page.extract_text(visitor_text=lambda s,cm,tm,f,fs:sizes.append(float(fs)) if s.strip() else None)
        assert min(sizes)>=5 and max(sizes)<=8,(stem,sorted(set(sizes)))
        fonts=[]
        for f in page['/Resources']['/Font'].values():
            f=f.get_object();name=str(f.get('/BaseFont'));desc=f['/DescendantFonts'][0].get_object()['/FontDescriptor'].get_object()
            assert 'Arial' in name and any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
            fonts.append(name)
        assert not any(v.get_object().get('/Subtype')=='/Image' for v in page['/Resources'].get('/XObject',{}).values())
        assert '<text' in (OUT/(stem+'.svg')).read_text(encoding='utf-8')
        modes={}
        for suffix in ['png','tiff']:
            with Image.open(OUT/(stem+'.'+suffix)) as im:
                assert im.mode=='RGB';dpi=[float(v) for v in im.info['dpi']];assert min(dpi)>599.9;im.verify();modes[suffix]=dpi
        assert not item.get('outside_text_candidates') and not item.get('text_bbox_overlaps')
        records.append(dict(stem=stem,width_mm=width,fonts=fonts,font_sizes_pt=sorted(set(sizes)),editable_vector=True,raster_dpi=modes))
    inputs=json.loads((QA/'figure_inputs.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in inputs.items())
    lock=json.loads((ROOT/'phase35_outputs/run_lock.json').read_text())
    for path,key in [('phase34_outputs/training_protocol.json','protocol_sha256'),('analysis_tools/mpra_fairness_training.py','source_sha256'),('phase34_inputs/architectures.py','architecture_sha256')]:assert sha(ROOT/path)==lock[key]
    # Independently compare plotted numerical rows to the preserved source tables.
    pts=pd.read_csv(QA/'plotted_values.csv',float_precision='round_trip')
    seeds=pd.read_csv(ROOT/'phase32_outputs/seed_metrics_pooled_and_fold.csv',float_precision='round_trip')
    cls=pd.read_csv(ROOT/'final_statistics/classification_summary.csv',float_precision='round_trip')
    tests=pd.read_csv(ROOT/'final_statistics/adjusted_tests.csv',float_precision='round_trip')
    gc=pd.read_csv(ROOT/'final_statistics/exact_gc_fold_metrics.csv',float_precision='round_trip')
    mpra=pd.read_csv(ROOT/'final_tables/main/T4_MPRA_internal.csv',float_precision='round_trip')
    seedm=pd.read_csv(ROOT/'phase37_outputs/seed_metrics_independent.csv',float_precision='round_trip')
    ext=pd.read_csv(ROOT/'final_tables/main/T5_External_transfer.csv',float_precision='round_trip')
    differences=[]
    def eq(a,b):differences.append(abs(float(a)-float(b)))
    for r in pts.itertuples():
        f=str(r.figure)
        if f=='5':
            if r.kind=='ensemble':v=cls[cls.stage.eq(r.stage)&cls.model.eq(r.model)].auroc.iloc[0]
            else:v=seeds[seeds.stage.eq(r.stage)&seeds.model.eq(r.model)&seeds.base_seed.eq(r.seed)].auroc.iloc[0]
            eq(r.auroc,v)
        elif f=='5c':
            s=tests[tests.comparison_id.eq(f'C_S{int(r.stage)}_DB2_minus_finetuned_dnabert_legacy')].iloc[0]
            for a,b in [('estimate','estimate_a_minus_b'),('lower','lower_95'),('upper','upper_95')]:eq(getattr(r,a),s[b])
        elif f in ['6a','6b']:
            g=gc[gc.stage.eq(r.stage)&gc.model.eq(r.model)];eq(r.raw,g.raw_fold_auroc.mean());eq(r.fold_sd,g.raw_fold_auroc.std(ddof=1));eq(r.exact_gc,g.same_gc_fold_auroc.mean())
        elif f in ['7a','7b']:
            g=seedm[seedm.cell.eq(r.cell)&seedm.cohort.eq('internal')&seedm.arm.eq(r.arm)&seedm.seed.eq(r.seed)] if r.kind=='seed' else mpra[mpra.cell.eq(r.cell)&mpra.arm.eq(r.arm)]
            assert len(g)==1;eq(r.pearson,g.pearson.iloc[0])
        elif f in ['7c','7d']:
            s=tests[tests.comparison_id.eq(r.comparison)].iloc[0]
            for a,b in [('estimate','estimate_a_minus_b'),('lower','lower_95'),('upper','upper_95'),('p_holm','p_holm')]:eq(getattr(r,a),s[b])
        elif f=='8':eq(r.value,ext[ext.cell.eq(r.cell)&ext.arm.eq(r.arm)].iloc[0][r.metric])
    assert max(differences)<1e-14
    # Approved Figure 1 masters remain byte-identical to the reviewed proposal.
    source=ROOT/'revision52_outputs/figure1_redesign_proposal_20261018_v1'
    for ext_,tail in [('pdf','.pdf'),('svg','.svg'),('png','_600dpi.png'),('tiff','_600dpi.tiff')]:assert sha(OUT/f'Figure1_Study_workflow.{ext_}')==sha(source/('Figure1_Study_workflow_proposed'+tail))
    docval=json.loads((QA/'docx_edit_validation.json').read_text());assert docval['prose_and_table_text_preserved'] and docval['figure_objects']==14
    result={'passed':True,'figure_filesets':14,'numeric_values_checked':len(differences),'maximum_numeric_difference':max(differences),'research_inputs_unchanged':True,'frozen_training_protocol_architecture_unchanged':True,'approved_figure1_identical':True,'vector_checks':records,'source_hashes':inputs}
    (QA/'asset_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['vector_checks','source_hashes']},indent=2))
if __name__=='__main__':validate()
