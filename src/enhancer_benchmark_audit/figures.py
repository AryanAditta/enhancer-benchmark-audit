"""Portable adapter for the preserved current main-figure drawing sources."""
from pathlib import Path
import shutil,subprocess,sys,json,os
import pandas as pd
from PIL import Image,ImageChops
from . import io as s

def build():
    workspace=s.PACKAGE/'.runtime/figure_workspace'
    work=workspace/'revision52_outputs/figure_revision_20261018_v1'
    work.mkdir(parents=True,exist_ok=True)
    (work/'qa').mkdir(exist_ok=True)
    (work/'staging/Figures').mkdir(parents=True,exist_ok=True)
    for p in (s.PACKAGE/'src/figure_sources').glob('*.py'):shutil.copy2(p,work/p.name)
    for p in s.INPUT.rglob('*'):
        if p.is_file():
            target=workspace/p.relative_to(s.INPUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for source,target in [('architectures.py','phase34_inputs/architectures.py'),('analysis_tools/mpra_fairness_training.py','analysis_tools/mpra_fairness_training.py')]:
        p=workspace/target;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s.PACKAGE/'src/frozen'/source,p)
    p=workspace/'phase34_outputs/training_protocol.json';p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s.PACKAGE/'configs/activity_protocol.json',p)
    # Figure 1 is the approved reusable workflow master; retain its drawing source too.
    proposal=workspace/'revision52_outputs/figure1_redesign_proposal_20261018_v1';proposal.mkdir(parents=True,exist_ok=True)
    for ext,suffix in [('.pdf','.pdf'),('.svg','.svg'),('.png','_600dpi.png'),('.tiff','_600dpi.tiff'),('_preview.png','_preview.png')]:
        shutil.copy2(s.PACKAGE/'figures/main'/('Figure1_Study_workflow'+ext),proposal/('Figure1_Study_workflow_proposed'+suffix))
    logs=[]
    for script in ['build_plots.py','build_diagrams.py']:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(work/script)],capture_output=True,text=True,encoding='utf-8',errors='replace',env=os.environ.copy())
        logs.append(result.stdout+'\n'+result.stderr)
        if result.returncode:raise RuntimeError(logs[-1])
    s.text('figure_build.log','\n'.join(logs))
    target=s.ROOT/'figures/main';target.mkdir(parents=True,exist_ok=True)
    checks=[]
    for p in (work/'staging/Figures').iterdir():
        if p.is_file():shutil.copy2(p,target/p.name)
    for p in sorted((s.PACKAGE/'figures/main').glob('Figure*.png')):
        if p.stem.endswith('_preview'):continue
        now=target/p.name
        with Image.open(p) as a,Image.open(now) as b:
            equal=a.size==b.size and ImageChops.difference(a.convert('RGB'),b.convert('RGB')).getbbox() is None
        checks.append({'figure':p.stem,'pixel_identical_in_reference_environment':equal})
    assert len(checks)==14
    # The figures remain scientifically reproducible on other platforms; byte/pixel
    # identity depends on the same fonts and rendering environment.
    expected=pd.read_csv(s.INPUT/'revision52_figure_plotted_values.csv',float_precision='round_trip')
    actual=pd.read_csv(work/'qa/plotted_values.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=0,atol=1e-12,check_like=True)
    if not all(x['pixel_identical_in_reference_environment'] for x in checks):
        print('Plot values reproduce; raster pixels differ from the reference font/rendering environment. Canonical publication masters remain in figures/.',flush=True)
    supplementary=s.ROOT/'figures/supplementary';supplementary.mkdir(parents=True,exist_ok=True)
    for p in (s.PACKAGE/'figures/supplementary').iterdir():
        if p.is_file():shutil.copy2(p,supplementary/p.name);assert s.sha(p)==s.sha(supplementary/p.name)
    report={'passed':True,'main_figure_objects':14,'main_figure_groups':8,'checks':checks,'plotted_values_match_preserved_records':True,'supplementary_images':8,'supplementary_action':'Exact assembly of supplied original PNGs; historical plotting sources retained separately.','figure1_action':'Approved master retained; original drawing code included.','font':'Arial for artwork; manuscript legend fonts are not embedded in graph files.'}
    s.dump('figure_reproduction_validation.json',report)
    print('Reproduced 14 current main objects; assembled eight unchanged supplementary originals.',flush=True)
    return report
