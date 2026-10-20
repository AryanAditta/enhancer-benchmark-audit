import numpy as np
import pandas as pd

def classification_metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, matthews_corrcoef
    y=np.asarray(y,int);p=np.asarray(p,float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(n=len(y),negative=int((y==0).sum()),positive=int(y.sum()),auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)),threshold=.5,sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),mcc=float(matthews_corrcoef(y,p>=.5)),balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def regression_metrics(y,p):
    from scipy.stats import pearsonr,spearmanr
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
    assert np.isfinite(y).all() and np.isfinite(p).all()
    return dict(n=len(y),pearson=float(pearsonr(y,p).statistic) if np.std(p)>0 and np.std(y)>0 else np.nan,spearman=float(spearmanr(y,p).statistic) if np.std(p)>0 and np.std(y)>0 else np.nan,rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),r2=float(1-np.dot(e,e)/np.sum((y-y.mean())**2)),mean_signed_error=float(e.mean()),prediction_sd=float(np.std(p)))

def weighted_corr(x,y,w):
    total=w.sum()
    if total<=1:return np.nan
    xx=x-np.dot(w,x)/total; yy=y-np.dot(w,y)/total
    den=np.sqrt(np.dot(w,xx*xx)*np.dot(w,yy*yy))
    return float(np.dot(w,xx*yy)/den) if den>0 else np.nan

def weighted_ranks(x,w):
    # Literal mean ranks of duplicated observations, including original ties.
    order=np.argsort(x,kind='stable');v=x[order];mass=w[order]
    starts=np.r_[0, np.flatnonzero(v[1:]!=v[:-1])+1]
    counts=np.add.reduceat(mass,starts)
    ranks=np.cumsum(counts)-counts+(counts+1)/2
    out=np.empty(len(x),float)
    out[order]=np.repeat(ranks,np.diff(np.r_[starts,len(x)]))
    return out

def weighted_auc(y,p,w):
    order=np.argsort(p,kind='stable');v=p[order];yy=y[order];mass=w[order]
    starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    positive=np.add.reduceat(mass*yy,starts);negative=np.add.reduceat(mass*(1-yy),starts)
    den=positive.sum()*negative.sum()
    return float(np.dot(positive,np.cumsum(negative)-negative+.5*negative)/den) if den else np.nan

def weighted_metric(y,p,w,metric):
    if metric=='auroc':return weighted_auc(y,p,w)
    if metric=='pearson':return weighted_corr(y,p,w)
    if metric=='spearman':return weighted_corr(weighted_ranks(y,w),weighted_ranks(p,w),w)
    raise ValueError(metric)

def stats_selfcheck():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr,pearsonr
    rng=np.random.default_rng(3091); checks=[]
    for i in range(25):
        y=rng.integers(0,2,12);p=np.round(rng.random(12),1);w=rng.integers(0,5,12)
        ix=np.repeat(np.arange(12),w);yy=y[ix];pp=p[ix]
        if len(np.unique(yy))<2:continue
        for metric,expected in [('auroc',roc_auc_score(yy,pp)),('pearson',pearsonr(yy,pp).statistic),('spearman',spearmanr(yy,pp).statistic)]:
            got=weighted_metric(y,p,w,metric);error=abs(got-expected)
            assert error<1e-12,(metric,error)
            checks.append(dict(probe=i,metric=metric,absolute_error=error,meaning='weighted evaluation versus literal duplicated rows with ties and zero weights'))
    return pd.DataFrame(checks)

def paired_arrays(row,classification,mpra):
    if row['task']=='classification':
        d=classification[classification.stage.eq(int(row['stage']))]
        score='score';arm='model';truth='label';group='similarity_group_id'
    else:
        d=mpra[mpra.cell.eq(row['cell'])&mpra.cohort.eq(row['cohort'])]
        score='prediction';arm='arm';truth='activity';group='group_id'
    a=d[d[arm].eq(row['arm_a'])].set_index('record_id').sort_index()
    b=d[d[arm].eq(row['arm_b'])].set_index('record_id').sort_index()
    assert a.index.equals(b.index) and np.array_equal(a[truth],b[truth]) and np.array_equal(a[group].astype(str),b[group].astype(str))
    groups,gi=np.unique(a[group].astype(str),return_inverse=True)
    return a[truth].to_numpy(),a[score].to_numpy(),b[score].to_numpy(),gi,len(groups)
