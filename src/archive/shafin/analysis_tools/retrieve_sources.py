from pathlib import Path
import urllib.request,hashlib,json,concurrent.futures,time
root=Path.cwd();dest=root/'reference_inputs/raw_sources'
specs={'hg38.2bit':('https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit','dcc3ea27079aa6dc3f9deccd7275e0f8'),'ENCFF774DYO.bed.gz':('https://www.encodeproject.org/files/ENCFF774DYO/@@download/ENCFF774DYO.bed.gz','87b3fb363f64aeb2b5f893ff98b47db3'),'ENCFF631LHA.bed.gz':('https://www.encodeproject.org/files/ENCFF631LHA/@@download/ENCFF631LHA.bed.gz','847512c673c75874e84200f84899f880')}
def get(item):
 name,(url,md5)=item;p=dest/name;tmp=dest/(name+'.part')
 if p.exists():
  assert hashlib.file_digest(p.open('rb'),'md5').hexdigest()==md5
  return name+' already authenticated'
 req=urllib.request.Request(url,headers={'User-Agent':'Enhancer-research-filtering-audit/1.0'})
 with urllib.request.urlopen(req,timeout=90) as resp,tmp.open('wb') as f:
  for b in iter(lambda:resp.read(8*1024*1024),b''):f.write(b)
 with tmp.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()==md5
 tmp.replace(p)
 return name+' downloaded and authenticated: '+str(p.stat().st_size)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for future in concurrent.futures.as_completed([pool.submit(get,x) for x in specs.items()]):
  try:print(future.result(),flush=True)
  except Exception as e:print('PENDING SOURCE',repr(e),flush=True)
