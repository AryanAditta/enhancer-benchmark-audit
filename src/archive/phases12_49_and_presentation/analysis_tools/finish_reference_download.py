from pathlib import Path
import subprocess,concurrent.futures,hashlib,json,time
r=Path.cwd().resolve();d=r/'reference_inputs/raw_sources';original=d/'hg38.2bit.part';start=original.stat().st_size;total=835393456;parts=d/'hg38_download_parts';parts.mkdir(exist_ok=True);url='https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit'
assert r in d.resolve().parents and r in parts.resolve().parents
specs=[(a,min(total-1,a+32*1024*1024-1)) for a in range(start,total,32*1024*1024)]
def fetch(spec):
 a,b=spec;p=parts/f'{a}_{b}.part'
 if p.exists() and p.stat().st_size==b-a+1:return p
 result=subprocess.run(['curl.exe','--fail','--silent','--show-error','--retry','2','--max-time','240','--range',f'{a}-{b}','--output',str(p),url],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
 if result.returncode or not p.exists() or p.stat().st_size!=b-a+1:raise RuntimeError(f'{a}-{b}: {result.stderr}; bytes {p.stat().st_size if p.exists() else 0}')
 print('Completed reference byte range',a,b,flush=True);return p
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:completed=list(pool.map(fetch,specs))
assembled=d/'hg38.2bit.assembling'
with assembled.open('wb') as out:
 for p in [original]+completed:
  with p.open('rb') as f:
   for data in iter(lambda:f.read(8*1024*1024),b''):out.write(data)
assert assembled.stat().st_size==total
with assembled.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='dcc3ea27079aa6dc3f9deccd7275e0f8'
assembled.replace(d/'hg38.2bit')
print('COMPLETE: full hg38 byte size and pinned MD5 verified',flush=True)
