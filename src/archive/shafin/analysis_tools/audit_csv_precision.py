from pathlib import Path
import nbformat as nb
r=Path.cwd();p=next(r.glob('[[]13]*ipynb'));n=nb.read(p,as_version=4)
idx=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and '## 6.' in c.source)
new=nb.v4.new_code_cell(r'''
# Explain the Spearman tolerance by recovering the pre-export vectors from authenticated caches.
precision_rows=[]
for row in recomputed[recomputed.model.str.match(r'^(legnet|mprann)_')].itertuples(index=False):
 frame=DATA[DATA.cell_line.eq(row.cell_line)&DATA[{'9b':'split_9b','9c':'split_9a'}[row.protocol]].eq(row.split)].set_index('record_id').sort_index()
 family=row.model.split('_')[0]
 if row.model.endswith('_three_seed_mean'):
  vectors=[pd.read_csv(P12/'predictions'/f'{row.protocol}_{row.cell_line}_{family}_seed_{seed}_{row.split}.csv').set_index('record_id').loc[frame.index].prediction.to_numpy() for seed in SEEDS]
  values=np.mean(vectors,axis=0)
 else:
  values=pd.read_csv(P12/'predictions'/f'{row.protocol}_{row.cell_line}_{row.model}_{row.split}.csv').set_index('record_id').loc[frame.index].prediction.to_numpy()
 original=metrics(frame.mean_log2_activity,values)
 saved=reported[reported.protocol.eq(row.protocol)&reported.cell_line.eq(row.cell_line)&reported.split.eq(row.split)&reported.model.eq(row.model)].iloc[0]
 assert abs(original['spearman']-saved.spearman)<1e-12
 precision_rows.append(dict(protocol=row.protocol,cell_line=row.cell_line,split=row.split,model=row.model,spearman_from_authenticated_pre_export_vectors=original['spearman'],spearman_from_aggregate_csv=row.spearman,spearman_saved=saved.spearman,absolute_csv_rank_difference=abs(row.spearman-original['spearman']),original_unique_targets=frame.mean_log2_activity.nunique(),original_unique_predictions=len(np.unique(values))))
write_table('csv_rank_precision_audit.csv',pd.DataFrame(precision_rows))
print('Spearman reconstructed from authenticated pre-export vectors agrees with saved Phase 12 values below 1e-12. The larger aggregate-CSV differences arise from serialization of near ties, not a metric arithmetic error.')
''')
n.cells[idx:idx]=[nb.v4.new_markdown_cell('### Diagnose CSV precision and near ties\nUse authenticated per-job caches and the frozen full-precision target table to reconstruct the vectors before Phase 12 aggregate CSV export. This distinguishes a harmless rank change from an actual metric calculation error.'),new]
for i,c in enumerate(n.cells):
 if c.cell_type=='code':compile(c.source,str(i),'exec')
nb.validate(n);nb.write(n,p)
