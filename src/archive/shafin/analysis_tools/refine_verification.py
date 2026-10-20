from pathlib import Path
import nbformat as nb
root=Path.cwd();p=next(root.glob('[[]13]*ipynb'));n=nb.read(p,as_version=4)
for c in n.cells:
 if c.cell_type=='code':
  c.source=c.source.replace("pred.model.str.contains('_seed_')", "pred.model.str.contains(r'_seed_\\d+$')")
  c.source=c.source.replace("assert (boots.upper_95<0).all()", "assert (boots[boots.contrast.str.startswith('dnabert2')].upper_95<0).all()")
  if 'seed_rows=[]' in c.source:
   c.source += "\nold_choices=csvmember('phase9c_validation_choices.csv')\nwrite_table('old_9c_validation_choices.csv',old_choices)\nfailed_choice=old_choices[old_choices.cell_line.eq('K562')&old_choices.model.eq('dnabert2_seed_20260930')].iloc[0]\nassert failed_choice.selected_epoch==1\nprint('Archived collapsed seed selected epoch 1. The two healthy seeds selected epoch 2. Keeping a near-constant seed contracts the ensemble scale; this explains why correlation barely changes while absolute errors increase, without identifying the training cause.')"
# Save detailed metric comparison, not just its maximum.
for c in n.cells:
 if c.cell_type=='code' and 'Largest metric differences' in c.source:
  c.source += "\nwrite_table('metric_recomputation_comparison.csv',check)"
extra=nb.v4.new_code_cell(r'''
# Link aggregate exports to per-job cache files and safely inspect checkpoint metadata on CPU.
import torch
checkpoint_audit=[]
for row in choices.itertuples(index=False):
 job=f'{row.protocol}_{row.cell_line}_{row.model}_seed_{row.seed}'
 ck=torch.load(P12/'checkpoints'/f'{job}_best.pt',map_location='cpu',weights_only=True)
 assert ck['lock']==prov['protocol_sha256'] and ck['job']==job and ck['epoch']==row.selected_epoch
 tr=DATA[DATA.cell_line.eq(row.cell_line)&DATA[{'9b':'split_9b','9c':'split_9a'}[row.protocol]].eq('train')].mean_log2_activity.to_numpy(np.float32)
 assert abs(ck['target_mean']-float(tr.mean()))<1e-9 and abs(ck['target_scale']-float(tr.std()))<1e-9
 for split in ['validation','test']:
  cached=pd.read_csv(P12/'predictions'/f'{job}_{split}.csv').set_index('record_id').sort_index()
  exported=pred[pred.protocol.eq(row.protocol)&pred.cell_line.eq(row.cell_line)&pred.split.eq(split)&pred.model.eq(f'{row.model}_seed_{row.seed}')].set_index('record_id').sort_index()
  assert cached.index.equals(exported.index) and np.max(abs(cached.prediction-exported.prediction))<1e-10
  if split=='validation':
   m=json.loads((P12/'cache'/f'{job}_validation.json').read_text());assert m['choice']['selected_epoch']==row.selected_epoch
   assert m['checkpoint_sha256']==prov['checkpoints'][f'{job}_best.pt']['sha256']
 cached=pd.read_csv(P12/'predictions'/f'{job}_external.csv').set_index('record_id').sort_index()
 assert cached.index.equals(ext.index) and np.max(abs(cached.prediction-newext[job]))<1e-10
 checkpoint_audit.append(dict(job=job,selected_epoch=ck['epoch'],target_mean=ck['target_mean'],target_scale=ck['target_scale'],checkpoint_metadata='verified_cpu_weights_only'))
 del ck
write_table('checkpoint_metadata_and_export_alignment.csv',pd.DataFrame(checkpoint_audit))
for col in ['cell_line','locus_group_id','split_group_id','split_9a','split_9b']:
 assert full.set_index('record_id')[col].sort_index().equals(splits.set_index('record_id')[col].sort_index()),col
print('24 checkpoints: selected epoch and train-only target scaling verified; all aggregate seed exports equal saved per-job scores.')
''')
idx=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and '## 5.' in c.source)
n.cells[idx:idx]=[nb.v4.new_markdown_cell('### Checkpoint metadata and export alignment\nCheckpoints are loaded as weights only on CPU. No model is instantiated or trained. This links selected epochs and train-only scaling to the actual saved checkpoints and links the aggregate prediction tables to their per-job caches.'),extra]
boot=next(c for c in n.cells if c.cell_type=='code' and 'def boot_difference' in c.source)
boot.source=r'''
from scipy.stats import rankdata
B=1000;rng=np.random.default_rng(20261013);boot_rows=[]
CONTRASTS=[('dnabert2_three_seed_mean','legnet_three_seed_mean'),('dnabert2_three_seed_mean','mprann_three_seed_mean'),('legnet_three_seed_mean','spaced6_4_ridge'),('mprann_three_seed_mean','spaced6_4_ridge'),('legnet_three_seed_mean','cnn_three_seed_mean')]
def group_drawer(labels):
 # Flatten groups and expand sampled group starts using vector arithmetic.
 _,codes=np.unique(labels,return_inverse=True);order=np.argsort(codes,kind='stable');lengths=np.bincount(codes);starts=np.r_[0,np.cumsum(lengths)[:-1]]
 def draw():
  picked=rng.integers(len(lengths),size=len(lengths));lens=lengths[picked]
  offsets=np.repeat(starts[picked],lens)+np.arange(lens.sum())-np.repeat(np.r_[0,np.cumsum(lens)[:-1]],lens)
  return order[offsets]
 return draw
def paired_repeats(y,wide,labels,stat):
 draw=group_drawer(labels);arms=sorted(set(a for pair in CONTRASTS for a in pair));arr=wide[arms].to_numpy();values=np.empty((B,len(CONTRASTS)))
 for j in range(B):
  idx=draw();yy=y[idx];pp=arr[idx]
  if stat=='spearman':yy=rankdata(yy);pp=rankdata(pp,axis=0)
  yy=yy-yy.mean();pp=pp-pp.mean(axis=0);rs=np.sum(yy[:,None]*pp,axis=0)/np.sqrt(np.sum(yy*yy)*np.sum(pp*pp,axis=0))
  for k,(a,b) in enumerate(CONTRASTS):values[j,k]=rs[arms.index(a)]-rs[arms.index(b)]
 assert np.isfinite(values).all()
 return np.quantile(values,[.025,.975],axis=0)
saved_ib=pd.read_csv(P12/'results/phase12_internal_paired_bootstrap.csv');saved_eb=pd.read_csv(P12/'results/phase12_external_paired_bootstrap.csv')
assert len(saved_ib)==20 and len(saved_eb)==40
assert (saved_ib.bootstrap_repeats==B).all() and (saved_eb.bootstrap_repeats==B).all()
assert (saved_ib.undefined_draws==0).all() and (saved_eb.undefined_draws==0).all()
dhs=ext[ext.origin.eq('DHS_natural')].copy();groups=pd.Series(index=dhs.index,dtype='int64');counter=-1
for chrom,g in dhs.sort_values(['external_chrom','external_start','external_end']).groupby('external_chrom'):
 boundary=-10**15
 for idx,row in g.iterrows():
  if row.external_start>boundary+1000:counter+=1
  boundary=max(boundary,row.external_end);groups.at[idx]=counter
print('External grouping:',len(dhs),'sequences;',groups.nunique(),'regions')
for tag,cell in itertools.product(['9b','9c'],['HepG2','K562']):
 sub=pred[pred.protocol.eq(tag)&pred.cell_line.eq(cell)&pred.split.eq('test')];w=sub.pivot(index='record_id',columns='model',values='prediction')
 truth=sub[sub.model.eq('spaced6_4_ridge')].set_index('record_id').loc[w.index];y=truth.target.to_numpy()
 ci=paired_repeats(y,w,truth.locus_group_id.to_numpy(),'pearson')
 for k,(a,b) in enumerate(CONTRASTS):
  contrast=f'{a} minus {b}';saved=saved_ib[saved_ib.protocol.eq(tag)&saved_ib.cell_line.eq(cell)&saved_ib.contrast.eq(contrast)].iloc[0]
  observed=corr(y,w[a])-corr(y,w[b]);assert abs(observed-saved.pearson_difference)<2e-8
  lo,hi=ci[:,k];assert max(lo,saved.lower_95)<=min(hi,saved.upper_95)
  boot_rows.append(dict(protocol=tag,cell_line=cell,contrast=contrast,cohort='internal_test',unit='locus',difference=observed,lower_95=lo,upper_95=hi,saved_lower_95=saved.lower_95,saved_upper_95=saved.upper_95,repeats=B))
 loc=ext.index.get_indexer(dhs.index);y=dhs[f'{cell}_l2fc'].to_numpy();ew=pd.DataFrame({arm:ext_scores[tag,cell,arm][loc] for arm in sorted(set(a for pair in CONTRASTS for a in pair))})
 for unit,gg,analysis in [('sequence',np.arange(len(y)),'primary'),('region_1kb',groups.to_numpy(),'sensitivity')]:
  ci=paired_repeats(y,ew,gg,'spearman')
  for k,(a,b) in enumerate(CONTRASTS):
   contrast=f'{a} minus {b}';saved=saved_eb[saved_eb.protocol.eq(tag)&saved_eb.cell_line.eq(cell)&saved_eb.contrast.eq(contrast)&saved_eb.analysis.eq(analysis)].iloc[0]
   observed=corr(rankdata(y),rankdata(ew[a]))-corr(rankdata(y),rankdata(ew[b]));assert abs(observed-saved.spearman_difference)<1e-5
   lo,hi=ci[:,k];assert max(lo,saved.lower_95)<=min(hi,saved.upper_95)
   boot_rows.append(dict(protocol=tag,cell_line=cell,contrast=contrast,cohort='DHS_natural',unit=unit,difference=observed,lower_95=lo,upper_95=hi,saved_lower_95=saved.lower_95,saved_upper_95=saved.upper_95,repeats=B))
 print('All five contrast bootstraps finished:',tag,cell,flush=True)
boots=pd.DataFrame(boot_rows);assert len(boots)==60;write_table('independent_paired_bootstrap.csv',boots)
print(boots.round(6).to_string(index=False))
'''
for c in n.cells:
 if c.cell_type=='markdown' and '## 5.' in c.source:c.source=c.source.replace('for both DNABERT-2 versus CNN contrasts','for all five reported contrasts (including both DNABERT-2 versus CNN contrasts)')
 if c.cell_type=='code' and 'report=dict(status=' in c.source:c.source=c.source.replace('bootstrap_seed=20261013,','bootstrap_seed=20261013,independently_repeated_intervals=len(boots),')
for i,c in enumerate(n.cells):
 if c.cell_type=='code':compile(c.source,str(i),'exec')
nb.validate(n);nb.write(n,p)
print('Extended notebook 13 to all 60 uncertainty intervals and checkpoint/export metadata.')
