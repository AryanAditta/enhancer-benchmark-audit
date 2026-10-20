"""Authenticated, resumable range downloads of pinned public pretrained weights."""
from pathlib import Path
import concurrent.futures,subprocess,json,re,hashlib,time,os,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'phase30_inputs/models'
def fsha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for model in ['dnabert2','dnabert_legacy']:
 folder=BASE/model;api=json.loads((folder/'model_api_revision.json').read_text());repo=api['id'];revision=api['sha']
 for spec in api['siblings']:
  name=spec['rfilename']
  if name in ['.gitattributes','pytorch_model.bin']:continue
  target=folder/name
  if target.exists() and target.stat().st_size==spec['size']:continue
  previous=ROOT/'reference_inputs/readiness_runtime/dnabert2_ec1f874'/name
  if model=='dnabert2' and previous.exists():shutil.copy2(previous,target);continue
  url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
  subprocess.run(['curl.exe','-L','-sS','--fail','--connect-timeout','15','--max-time','90','-o',str(target),url],check=True)
  assert target.stat().st_size==spec['size'],name
 spec=next(x for x in api['siblings'] if x['rfilename']=='pytorch_model.bin');size=spec['size'];expected=spec['lfs']['sha256'];target=folder/'pytorch_model.bin'
 if target.exists() and target.stat().st_size==size and fsha(target)==expected:continue
 chunks=folder/'weight_ranges';chunks.mkdir(exist_ok=True);step=8*1024*1024
 ranges=[(start,min(start+step,size)-1) for start in range(0,size,step)]
 def download(bounds):
  start,end=bounds;path=chunks/f'{start:012d}_{end:012d}.bin';meta=path.with_suffix('.json')
  if path.exists() and meta.exists():
   saved=json.loads(meta.read_text())
   if path.stat().st_size==end-start+1 and fsha(path)==saved['sha256']:return path
  temp=path.with_suffix('.downloading');headers=path.with_suffix('.headers')
  url=f'https://huggingface.co/{repo}/resolve/{revision}/pytorch_model.bin?download=true&part={start}'
  for attempt in range(4):
   p=subprocess.run(['curl.exe','-L','-sS','--fail','--connect-timeout','20','--max-time','180','--range',f'{start}-{end}',
    '-D',str(headers),'-o',str(temp),url],capture_output=True,text=True)
   h=headers.read_text(encoding='utf8',errors='replace') if headers.exists() else ''
   spans=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',h,re.I)
   if p.returncode==0 and spans and tuple(map(int,spans[-1]))==(start,end,size) and temp.stat().st_size==end-start+1:
    os.replace(temp,path);meta.write_text(json.dumps({'start':start,'end':end,'total':size,'sha256':fsha(path),'revision':revision},indent=2));return path
   print(model,'range',start,'attempt',attempt+1,'not complete',p.returncode,flush=True);time.sleep(2)
  raise RuntimeError((model,start,end,'range download failed'))
 # Valid existing8MB probe provides a useful cached range after full header verification.
 probe=folder/'range_probe_8mb.bin';probeheaders=folder/'range_probe_headers.txt'
 # Its offset1MB differs from the8MB grid, so it stays a diagnostic and is not spliced blindly.
 finished=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(download,b):b for b in ranges}
  for f in concurrent.futures.as_completed(futures):
   finished.append(f.result())
   status={'model':model,'revision':revision,'completed_ranges':len(finished),'total_ranges':len(ranges),'completed_bytes':sum(p.stat().st_size for p in finished)}
   (BASE/'download_progress.json').write_text(json.dumps(status,indent=2)+'\n')
   print(model,len(finished),'/',len(ranges),'ranges saved',flush=True)
 assembled=target.with_suffix('.assembled')
 with assembled.open('wb') as w:
  for start,end in ranges:
   with (chunks/f'{start:012d}_{end:012d}.bin').open('rb') as rd:shutil.copyfileobj(rd,w,8*1024*1024)
 assert assembled.stat().st_size==size and fsha(assembled)==expected,(model,'assembled hash')
 os.replace(assembled,target);print('AUTHENTICATED',model,target.stat().st_size,expected,flush=True)
print('Both pinned pretrained snapshots downloaded and authenticated.',flush=True)
