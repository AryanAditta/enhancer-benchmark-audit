from pathlib import Path
import nbformat as nb
root=Path.cwd();p=next(root.glob('[[]14]*ipynb'));n=nb.read(p,as_version=4)
n.cells.append(nb.v4.new_markdown_cell('## 5. Export reusable population figures\nFigures are descriptive and keep both cell/design families separate. The subgroup analysis above uses retained test records only.'))
n.cells.append(nb.v4.new_code_cell(r'''
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if POP is not None:
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 colors={'retained':'#2364AA','removed':'#D97732'}
 for row,cell in enumerate(['HepG2','K562']):
  sub=POP[POP.cell_line.eq(cell)]
  for col,feature in enumerate(['mean_log2_activity','gc_fraction']):
   bins=np.linspace(sub[feature].min(),sub[feature].max(),55)
   for population,g in sub.groupby('population'):
    axes[row,col].hist(g[feature],bins=bins,weights=np.ones(len(g))/len(g),histtype='step',linewidth=1.8,color=colors[population],label=f'{population} (n={len(g):,})')
   axes[row,col].set_title(cell+' — '+('assayed activity' if col==0 else 'GC composition'))
   axes[row,col].set_xlabel('Mean log2 output/input' if col==0 else 'GC fraction');axes[row,col].set_ylabel('Fraction of population per bin');axes[row,col].legend(frameon=False,fontsize=8)
 fig.suptitle('Selection effects of the legacy-similarity screen',fontsize=14)
 fig.savefig(OUT/'retained_removed_activity_and_gc.png',dpi=220);fig.savefig(OUT/'retained_removed_activity_and_gc.pdf');plt.close(fig)
 print('Saved population comparison figure as PNG and PDF.')
else:print('Population figures pending complete authenticated reference copy.')
'''))
# Explain absolute-error differences for the old collapsed-seed ensemble without changing it.
for c in n.cells:
 if c.cell_type=='markdown' and '## 3.' in c.source:c.source+='\n\nThe Phase 9C K562 DNABERT-2 ensemble includes the archived nearly constant seed. Its scale contraction can worsen absolute-error subgroups; notebook 13 reports a labelled healthy-seed sensitivity. Activity subgroups require measured labels and are diagnostic summaries, not deployment-time routing rules.'
nb.validate(n)
for i,c in enumerate(n.cells):
 if c.cell_type=='code':compile(c.source,str(i),'exec')
nb.write(n,p)
print('Added population figures and the collapsed-ensemble subgroup caveat.')
