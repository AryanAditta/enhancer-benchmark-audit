from pathlib import Path
import os,sys,json,nbformat,time
root=Path.cwd();runtime=root/'analysis_tools/runtime';runtime.mkdir(exist_ok=True)
for var in ['TEMP','TMP','JUPYTER_RUNTIME_DIR','IPYTHONDIR','MPLCONFIGDIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR']:
 p=runtime/var.lower();p.mkdir(exist_ok=True);os.environ[var]=str(p)
kdir=root/'analysis_tools/kernels/enhancer';kdir.mkdir(parents=True,exist_ok=True)
(kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-B','-X','utf8','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python (enhancer)','language':'python'}),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(root/'analysis_tools')
from nbclient import NotebookClient
number=int(sys.argv[1]);path=next(root.glob(f'[[]{number}]*.ipynb'));n=nbformat.read(path,as_version=4)
for i,c in enumerate(n.cells):
 if c.cell_type=='code':compile(c.source,f'{path.name}:cell{i}','exec')
nbformat.validate(n);out=root/f'phase{number}_outputs';out.mkdir(exist_ok=True);log=out/'execution_log.txt';start=time.time()
def progress(cell,cell_index,**kwargs):
 text=f'Completed cell {cell_index} in {time.time()-start:.1f}s'
 print(text,flush=True)
 nbformat.write(n,path)
 with log.open('a',encoding='utf-8') as f:f.write(text+'\n')
log.write_text('Executed with '+sys.executable+'\n',encoding='utf-8')
client=NotebookClient(n,timeout=int(os.environ.get('ENHANCER_CELL_TIMEOUT','21600')),kernel_name='enhancer',resources={'metadata':{'path':str(root)}},on_cell_executed=progress)
try:client.execute()
finally:
 nbformat.validate(n);nbformat.write(n,path)
 errors=[o for c in n.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
 print('Notebook',number,'saved; errors:',len(errors),'elapsed',round(time.time()-start,1),flush=True)
