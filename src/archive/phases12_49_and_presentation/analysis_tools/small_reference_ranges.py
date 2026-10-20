from pathlib import Path
import subprocess,concurrent.futures,hashlib,time
r=Path.cwd().resolve();d=r/'reference_inputs/raw_sources';initial=d/'hg38.2bit.part';start=initial.stat().st_size;total=835393456;parts=d/'hg38_small_ranges';parts.mkdir(exist_ok=True);url='https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit';specs=[(a,min(total-1,a+1024*1024-1)) for a in range(start,total,1024*1024)]
def fetch(spec):
 a,b=spec;p=parts/f'{a}_{b}.part'
 if p.exists() and p.stat().st_size==b-a+1:return p
 for retry in range(3):
  result=subprocess.run(['curl.exe','--fail','--silent','--show-error','--max-time','90','--range',f'{a}-{b}','--output',str(p),url],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
  if result.returncode==0 and p.exists() and p.stat().st_size==b-a+1:return p
 raise RuntimeError(f'{a}-{b}: {result.stderr}')
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
 futures={pool.submit(fetch,x):i for i,x in enumerate(specs)};completed={}
 for future in concurrent.futures.as_completed(futures):
  completed[futures[future]]=future.result()
  if len(completed)%20==0:print('Completed',len(completed),'of',len(specs),'small reference ranges',flush=True)
assembled=d/'hg38.2bit.assembling_small'
with assembled.open('wb') as out:
 for p in [initial]+[completed[i] for i in range(len(specs))]:
  with p.open('rb') as f:
   for data in iter(lambda:f.read(8*1024*1024),b''):out.write(data)
assert assembled.stat().st_size==total
with assembled.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='dcc3ea27079aa6dc3f9deccd7275e0f8'
assembled.replace(d/'hg38.2bit');print('COMPLETE reference; byte size and pinned MD5 authenticated',flush=True)
