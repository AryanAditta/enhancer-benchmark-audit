import sys,itertools,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_tools import final_synthesis as s
def lcs(a,b):
    masks={c:0 for c in 'ACGT'}
    for i,c in enumerate(a):masks[c]|=1<<i
    state=0
    for c in b:
        x=masks[c]|state;state=x&~(x-((state<<1)|1))
    return state.bit_count()
def rc(x):return x.translate(str.maketrans('ACGT','TGCA'))[::-1]
s.verify_inputs();rows=[]
for stage,m in zip([1,2],s.manifests()):
    candidate=m[m.role.eq('train_candidate')];assert candidate.groupby('similarity_group_id').fold_5.nunique().max()==1
    canonical=candidate.sequence.map(lambda x:min(x,rc(x)));assert candidate.assign(canonical=canonical).groupby('canonical').fold_5.nunique().max()==1
    checked=0
    if '--exhaustive' in sys.argv:
        # Recompute every pair among retained candidates. CPU-intensive; no training.
        records=list(candidate.itertuples())
        for a,b in itertools.combinations(records,2):
            if max(lcs(a.sequence,b.sequence),lcs(a.sequence,rc(b.sequence)))>=160:
                assert a.similarity_group_id==b.similarity_group_id
            checked+=1
    rows.append(dict(stage=stage,candidate_records=len(candidate),groups=candidate.similarity_group_id.nunique(),fold_isolation_passed=True,exhaustive_pairs_checked=checked))
# Independently recompute the archived positive Stage1 LCS edge values.
m=s.manifests()[0].drop_duplicates('record_id').set_index('record_id');edges=pd.read_csv(s.PACKAGE/'inputs/final_evidence/inputs/stage1_lcs_ge150_recomputed.tsv',sep='\t');edge_checks=0
for row in edges.itertuples():
    if row.id_a not in m.index or row.id_b not in m.index:continue
    a=m.loc[row.id_a,'sequence'];b=m.loc[row.id_b,'sequence'];assert lcs(a,b)==row.lcs_forward and lcs(a,rc(b))==row.lcs_rc;edge_checks+=1
s.dump('grouping_validation.json',dict(passed=True,tasks=rows,positive_edges_recomputed=edge_checks,full_pair_rescan=('--exhaustive' in sys.argv),limitation='Default test verifies fixed-group fold isolation, exact/RC pairs and supplied positive edges; absence of additional edges requires --exhaustive. This does not rule out every biological relationship.'))
print('Fixed grouping isolation and positive LCS edge checks passed.')
