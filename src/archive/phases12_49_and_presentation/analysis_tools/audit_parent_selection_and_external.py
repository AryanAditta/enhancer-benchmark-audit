from pathlib import Path
import nbformat as nb
r=Path.cwd();p=next(r.glob('[[]13]*ipynb'));n=nb.read(p,as_version=4)
idx=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and '## 6.' in c.source)
c=nb.v4.new_code_cell(r'''
parent_hashes=[];older_choices=[]
for tag in ['9b','9c','10']:
 parent=json.loads(member(f'phase{tag}_provenance.json'));assert parent['protocol_sha256']==prov['config']['parent_locks'][tag]
 for name,h in parent['result_sha256'].items():
  assert digest(member(name))==h,name
  parent_hashes.append(dict(protocol=tag,file=name,sha256=h))
 if tag!='10':
  choices_old=csvmember(f'phase{tag}_validation_choices.csv')
  for row in choices_old.itertuples(index=False):
   got=recomputed[recomputed.protocol.eq(tag)&recomputed.cell_line.eq(row.cell_line)&recomputed.split.eq('validation')&recomputed.model.eq(row.model)].iloc[0]
   assert abs(got.pearson-row.validation_pearson)<1e-6
  choices_old['protocol']=tag;choices_old['audit_status']='Saved validation score verified; per-epoch maxima not independently provable without original training logs';older_choices.append(choices_old)
write_table('parent_result_hash_checks.csv',pd.DataFrame(parent_hashes));write_table('older_validation_selection_audit.csv',pd.concat(older_choices,ignore_index=True))
# Independently check external exact/RC identity and natural genomic interval separation.
COMP=str.maketrans('ACGT','TGCA')
def canonical(s):return min(s,s.translate(COMP)[::-1])
refcanonical={canonical(s) for s in DATA.sequence};externalcanonical=ext.sequence.map(canonical)
assert externalcanonical.is_unique and not externalcanonical.isin(refcanonical).any()
interval_rows=[]
for origin,g in ext[ext.origin.isin(['DHS_natural','Malinois_natural'])].groupby('origin'):
 overlap=0
 for chrom,part in g.groupby('external_chrom'):
  loci=DATA[DATA.chrom.eq(chrom)].sort_values('start');starts=loci.start.to_numpy();maxends=np.maximum.accumulate(loci.end.to_numpy())
  st=part.external_start.to_numpy();en=part.external_end.to_numpy();assert (en-st==200).all()
  ix=np.searchsorted(starts,en,side='left')-1;valid=ix>=0
  overlap+=int(np.sum(maxends[ix[valid]]>st[valid]))
 assert overlap==0,(origin,overlap)
 interval_rows.append(dict(origin=origin,n=len(g),interval_intersections_with_phase9=overlap))
write_table('external_natural_interval_separation.csv',pd.DataFrame(interval_rows))
# Canonical 31-mers use two-bit integers, avoiding an enormous Python string set.
K=31;MASK=np.uint64((1<<(2*K))-1);lut=np.zeros(256,dtype=np.uint64)
for j,b in enumerate(b'ACGT'):lut[b]=j

def canonical_windows(sequences):
 codes=lut[np.stack([np.frombuffer(s.encode(),dtype=np.uint8) for s in sequences])]
 f=np.zeros(len(codes),dtype=np.uint64);v=f.copy();windows=[]
 for i in range(200):
  f=((f<<np.uint64(2))|codes[:,i])&MASK
  v=(v>>np.uint64(2))|((np.uint64(3)-codes[:,i])<<np.uint64(2*(K-1)))
  if i>=K-1:windows.append(np.minimum(f,v))
 return np.stack(windows,axis=1)
reference_words=np.unique(canonical_windows(DATA.sequence.drop_duplicates().tolist()).ravel())
audit10=json.loads(member('phase10_inclusion_audit.json'));assert len(reference_words)==audit10['phase9_unique_canonical_31mers']
hits=[]
for start in range(0,len(ext),1000):
 windows=canonical_windows(ext.sequence.iloc[start:start+1000].tolist());ix=np.searchsorted(reference_words,windows)
 valid=ix<len(reference_words);match=np.zeros(windows.shape,dtype=bool);match[valid]=reference_words[ix[valid]]==windows[valid]
 hits.extend(match.any(axis=1).tolist())
assert not any(hits)
dump_json('external_sequence_separation_audit.json',dict(external_n=29330,exact_or_rc_matches_to_phase9=0,shared_canonical_31mer_records=0,reference_unique_canonical_31mers=len(reference_words),natural_interval_overlap=interval_rows,limitations='Exact/RC and 31-mer checks do not prove absence of all homology; natural sequences remain selected experimental libraries.'))
del reference_words
print('15 parent result hashes and 32 older validation choices checked; older epoch maxima require missing training logs. External records: no exact/RC identity, natural interval overlap or shared canonical 31-mer with Phase 9A.')
''')
n.cells[idx:idx]=[nb.v4.new_markdown_cell('### Upstream selections and external sequence separation\nOlder validation-choice scores can be checked against their saved predictions. Their epoch maxima cannot be independently established without the original per-epoch logs; this remains an explicit limitation. Separately, authenticate parent results and check the frozen external cohort for exact/RC identity, genomic interval overlap and canonical 31-mer sharing with all Phase 9A sequences.'),c]
for cell in n.cells:
 if cell.cell_type=='code' and 'report=dict(status=' in cell.source:cell.source=cell.source.replace("old_9c_k562_collapse=", "older_epoch_maxima='Unverified: original per-epoch logs unavailable; reported validation scores checked',external_separation='No exact/RC, 31-mer or natural interval matches to Phase 9A',old_9c_k562_collapse=")
for i,cell in enumerate(n.cells):
 if cell.cell_type=='code':compile(cell.source,str(i),'exec')
nb.validate(n);nb.write(n,p)
