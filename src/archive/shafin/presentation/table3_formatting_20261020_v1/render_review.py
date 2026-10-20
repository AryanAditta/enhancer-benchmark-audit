from pathlib import Path
import os, sys, tempfile, subprocess, importlib.util, json
ROOT = Path(r"D:\Aditta Thesis\For Shafin\Shafin's Contribution")
WORK = Path(__file__).resolve().parent
DEPS = Path(r'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies')
for key in ['TEMP','TMP','TMPDIR','XDG_CACHE_HOME']:
    directory = WORK/'runtime'/key.lower()
    directory.mkdir(parents=True,exist_ok=True)
    os.environ[key] = str(directory)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
tempfile.tempdir=os.environ['TMP']
sys.path.append(str(DEPS/'python/Lib/site-packages'))
os.environ['PATH'] = str(DEPS/'native/poppler/Library/bin') + os.pathsep + os.environ['PATH']
renderer_path = Path(r'C:\Users\User\.codex\plugins\cache\openai-primary-runtime\documents\26.1015.12040\skills\documents\render_docx.py')
spec=importlib.util.spec_from_file_location('table3_renderer',renderer_path)
renderer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
doc=WORK/'staging/Manuscript.docx'
out=WORK/'qa/render'
out.mkdir(parents=True,exist_ok=True)
pdf=out/'Manuscript.pdf'
cmd=[r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-NonInteractive','-File',str(ROOT/'analysis_tools/export_final_docx_word.ps1'),'-InputPath',str(doc),'-OutputPath',str(pdf)]
result=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240,creationflags=subprocess.CREATE_NO_WINDOW)
(WORK/'qa/render.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
if result.returncode: raise RuntimeError(result.stderr)
renderer.convert_to_pdf=lambda *args,**kwargs: (str(pdf),'Native Word read-only PDF export; bundled LibreOffice unavailable on this Windows runtime')
pages=renderer.rasterize(str(doc),str(out),150,False,False)
from pypdf import PdfReader
from PIL import Image, ImageChops
old=ROOT/'revision52_outputs/table2_formatting_20261020_v1/qa/render'
reader=PdfReader(pdf)
table_pages=[i+1 for i,page in enumerate(reader.pages) if 'A. Supplied files' in (page.extract_text() or '') or 'B. Retained candidates after historical-test priority and related-group exclusions' in (page.extract_text() or '')]
unchanged=[]
changed=[]
for i in range(1,len(pages)+1):
    before=old/f'page-{i}.png'
    now=out/f'page-{i}.png'
    if before.is_file():
        with Image.open(before) as a, Image.open(now) as b:
            equal=a.size==b.size and ImageChops.difference(a.convert('RGB'),b.convert('RGB')).getbbox() is None
        (unchanged if equal else changed).append(i)
report={'pages':len(pages),'table3_pages':table_pages,'unchanged_page_images':unchanged,'changed_page_images':changed,'backend':'Native Word read-only export followed by managed render_docx.py and bundled Poppler'}
(WORK/'qa/render_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
