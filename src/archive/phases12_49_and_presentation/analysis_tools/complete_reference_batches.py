from pathlib import Path
import subprocess,concurrent.futures,hashlib,json,time
r=Path.cwd().resolve();d=r/'reference_inputs/raw_sources';initial=d/'hg38.2bit.part';start=initial.stat().st_size;total=835393456;parts=d/'hg38_small_ranges';parts.mkdir(exist_ok=True);url='https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit';specs=[(a,min(total-1,a+1024*1024-1)) for a in range(start,total,1024*1024)]
def path(spec):return parts/f'{spec[0]}_{spec[1]}.part'
def missing():return [s for s in specs if not path(s).exists() or path(s).stat().st_size!=s[1]-s[0]+1]
batches=r/'analysis_tools/runtime/download_batches';batches.mkdir(exist_ok=True)
for attempt in range(3):
 todo=missing();print('Reference missing chunks:',len(todo),'attempt',attempt+1,flush=True)
 if not todo:break
 configs=[]
 for worker in range(6):
  items=todo[worker::6]
  if not items:continue
  cfg=batches/f'round{attempt}_worker{worker}.cfg';blocks=[]
  for spec in items:
   a,b=spec;blocks.append('\n'.join(['fail','silent','show-error','retry = 3','retry-all-errors','retry-delay = 2','max-time = 60',f'url = "{url}"',f'range = "{a}-{b}"',f'output = "{path(spec).as_posix()}"']))
  cfg.write_text('\nnext\n'.join(blocks)+'\n',encoding='utf-8');configs.append(cfg)
 def batch(cfg):
  result=subprocess.run(['curl.exe','--config',str(cfg)],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
  (cfg.with_suffix('.log')).write_text(result.stderr,encoding='utf-8');print('Finished download batch',cfg.name,'; remaining',len(missing()),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(batch,configs))
assert not missing(),f'{len(missing())} reference chunks still missing'
assembled=d/'hg38.2bit.assembling'
with assembled.open('wb') as out:
 for p in [initial]+[path(s) for s in specs]:
  with p.open('rb') as f:
   for data in iter(lambda:f.read(8*1024*1024),b''):out.write(data)
assert assembled.stat().st_size==total
with assembled.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='dcc3ea27079aa6dc3f9deccd7275e0f8'
assembled.replace(d/'hg38.2bit');print('COMPLETE: exact reference size and pinned MD5 authenticated',flush=True)
