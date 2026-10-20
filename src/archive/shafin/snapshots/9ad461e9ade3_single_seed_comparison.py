from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis_tools import final_synthesis as s
s.verify_inputs();p=pd.read_csv(s.PACKAGE/'predictions/classification_current_all_seeds.csv.gz',float_precision='round_trip');expected=pd.read_csv(s.PACKAGE/'tables/Single_seed_and_ensemble_comparison.csv',float_precision='round_trip');rows=[]
for (stage,model),g in p.groupby(['stage','model']):
    model={'dnabert2':'finetuned_dnabert2','dnabert_legacy':'finetuned_dnabert_legacy'}.get(model,model)
    values=[roc_auc_score(h.label,h.score) for _,h in g.groupby('base_seed')]
    r=expected[expected.stage.eq(stage)&expected.model.eq(model)].iloc[0]
    for got,want in [(np.mean(values),r.single_seed_pooled_AUROC_mean),(np.std(values,ddof=1),r.single_seed_pooled_AUROC_SD),(np.mean(values)-r.spaced_mer_single_pooled_AUROC,r.single_seed_mean_minus_spaced_mer)]:assert abs(got-want)<1e-12
    rows.append(dict(stage=int(stage),model=model,single_seed_mean=float(np.mean(values)),single_seed_SD=float(np.std(values,ddof=1)),single_seed_minus_spaced_mer=float(np.mean(values)-r.spaced_mer_single_pooled_AUROC)))
s.dump('single_seed_validation.json',dict(passed=True,rows=rows,summaries=4,aggregation='pooled OOF per seed, not mean-fold',new_hypothesis_tests=0))
print('All four single-seed summaries reproduced from saved predictions.')
