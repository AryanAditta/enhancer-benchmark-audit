from pathlib import Path
import urllib.request, hashlib, json, concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reference_inputs/phase19_method_sources';DEST.mkdir(exist_ok=True)
SOURCES={
 'legnet_LICENSE.txt':'https://raw.githubusercontent.com/autosome-ru/human_legnet/a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d/LICENSE',
 'mprann_LICENSE.txt':'https://raw.githubusercontent.com/visze/sequence_cnn_models/7638fce137db0445123efe8d7e2c35e248fafc5f/LICENSE',
 'conformal_beyond_exchangeability.html':'https://arxiv.org/html/2202.13415v5',
 'dnabert2_primary_abstract.html':'https://arxiv.org/abs/2306.15006',
 'human_legnet_README.md':'https://raw.githubusercontent.com/autosome-ru/human_legnet/a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d/README.md',
 'mprann_README.md':'https://raw.githubusercontent.com/visze/sequence_cnn_models/7638fce137db0445123efe8d7e2c35e248fafc5f/README.md',
}
def get(item):
 name,url=item;p=DEST/name
 try:
  if p.exists():raw=p.read_bytes()
  else:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Enhancer-research-method-audit/1.0'}),timeout=30) as response:raw=response.read()
   p.write_bytes(raw)
  return {'name':name,'url':url,'status':'saved','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 except Exception as e:return {'name':name,'url':url,'status':'retrieval_unavailable','error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(get,SOURCES.items()))
(DEST/'source_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
for r in rows:print(r['name'],r['status'],r.get('bytes',''),flush=True)
