from figure_common import *
import sys, importlib.util, subprocess, tempfile
DEPS=Path(r'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies')
sys.path.append(str(DEPS/'python/Lib/site-packages'))
os.environ['PATH']=str(DEPS/'native/poppler/Library/bin')+os.pathsep+os.environ['PATH']
tempfile.tempdir=os.environ['TMP']
renderer_path=Path(r'C:\Users\User\.codex\plugins\cache\openai-primary-runtime\documents\26.1015.12040\skills\documents\render_docx.py')
spec=importlib.util.spec_from_file_location('figure_revision_renderer',renderer_path);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
doc=WORK/'staging/Manuscript.docx';out=QA/'Manuscript_render_final';out.mkdir(exist_ok=True)
assert doc.is_file(),doc
status('Rendering revised manuscript for full-page visual inspection')
# The bundled runtime has no LibreOffice on Windows. Do not launch installed desktop LO.
def missing_bundled_lo():raise FileNotFoundError('Bundled LibreOffice is unavailable on this Windows runtime')
renderer._resolve_soffice=missing_bundled_lo
try:
    pages=renderer.rasterize(str(doc),str(out),150,False,True)
    backend='bundled LibreOffice'
except FileNotFoundError:
    pdf=out/'Manuscript.pdf'
    cmd=[r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-NonInteractive','-File',str(ROOT/'analysis_tools/export_final_docx_word.ps1'),'-InputPath',str(doc),'-OutputPath',str(pdf)]
    result=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240,creationflags=subprocess.CREATE_NO_WINDOW)
    (QA/'word_render.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr)
    renderer.convert_to_pdf=lambda *args,**kwargs:(str(pdf),'Native Word read-only PDF export')
    pages=renderer.rasterize(str(doc),str(out),150,False,False)
    backend='native Word PDF export and bundled render_docx.py / Poppler'
(QA/'render_backend.json').write_text(json.dumps({'backend':backend,'renderer_sha256':sha(renderer_path),'docx_sha256':sha(doc),'pages':len(pages),'settings_changed':False},indent=2),encoding='utf-8')
print('Rendered manuscript pages',len(pages))
