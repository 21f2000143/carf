#!/usr/bin/env python3
"""Prespecified MRI + transcriptomic + clinical fusion analysis."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/revision2/processed'; OUT=ROOT/'results/revision2'
SEED=20260917

def metrics(y,p):
 return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':brier_score_loss(y,p)}
def rank_normalize(a):
 a=np.asarray(a,float)
 if a.ndim!=2: raise ValueError('Expression must be a two-dimensional matrix')
 ranked=np.full(a.shape,np.nan)
 for i,row in enumerate(a):
  observed=np.isfinite(row)
  if observed.any(): ranked[i,observed]=rankdata(row[observed],method='average')/observed.sum()
 return ranked
def common_data():
 m2=pd.read_csv(DATA/'ISPY2_meta.csv',dtype={'patient_id':str}).set_index('patient_id')
 m1=pd.read_csv(DATA/'ISPY1_meta.csv',dtype={'patient_id':str}).set_index('patient_id')
 e2=pd.read_csv(DATA/'ISPY2_expression.csv.gz',index_col=0); e2.index=e2.index.astype(str)
 e1=pd.read_csv(DATA/'ISPY1_expression.csv.gz',index_col=0); e1.index=e1.index.astype(str)
 i2=pd.read_csv(DATA/'ISPY2_mri.csv',index_col=0); i1=pd.read_csv(DATA/'ISPY1_mri.csv',index_col=0); i2.index=i2.index.astype(str); i1.index=i1.index.astype(str)
 ids2=m2.index.intersection(e2.index).intersection(i2.index); ids1=m1.index.intersection(e1.index).intersection(i1.index)
 genes=e2.columns.intersection(e1.columns)
 # Patient-local ranks make the two public assay platforms comparable without external moments.
 E2=pd.DataFrame(rank_normalize(e2.loc[ids2,genes].to_numpy()),index=ids2,columns=genes)
 E1=pd.DataFrame(rank_normalize(e1.loc[ids1,genes].to_numpy()),index=ids1,columns=genes)
 # The assay intersection is label-free; variance filtering belongs inside each fit.
 return E2,m2.loc[ids2],i2.loc[ids2],E1,m1.loc[ids1],i1.loc[ids1]

class TopVariance(BaseEstimator, TransformerMixin):
 """Select high-variance genes using this fit's training patients only."""
 def __init__(self, k=3000): self.k=k
 def fit(self, X, y=None):
  self.indices_=np.argsort(-np.var(np.asarray(X),axis=0),kind='stable')[:self.k]
  return self
 def transform(self, X): return np.asarray(X)[:,self.indices_]

def fit_expert(X,y,kind,seed=SEED):
 if kind=='rna':
  return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),TopVariance(3000),SelectKBest(f_classif,k=min(250,X.shape[1])),StandardScaler(),LogisticRegression(C=0.1,class_weight='balanced',max_iter=1000,solver='liblinear')).fit(X,y)
 if kind=='clinical':
  cols=[c for c in ['age','hr','her2'] if c in X.columns]
  return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,solver='liblinear')).fit(X[cols],y)
 return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,solver='liblinear')).fit(X,y)
def predict_expert(model,X,kind):
 if kind=='clinical': X=X[[c for c in ['age','hr','her2'] if c in X.columns]]
 return model.predict_proba(X)[:,1]
def experts(E,M,I,y,seed=SEED):
 return [fit_expert(E,y,'rna',seed),fit_expert(M,y,'clinical',seed),fit_expert(I,y,'mri',seed)]
def oof(E,M,I,y,seed=SEED):
 p=np.zeros((len(y),3)); cv=StratifiedKFold(5,shuffle=True,random_state=seed)
 for tr,te in cv.split(E,y):
  for j,(X,k) in enumerate([(E,'rna'),(M,'clinical'),(I,'mri')]):
   p[te,j]=predict_expert(fit_expert(X.iloc[tr],y[tr],k,seed),X.iloc[te],k)
 return p

def run():
 E2,M2,I2,E1,M1,I1=common_data(); y2=M2.pcr.astype(int).to_numpy(); y1=M1.pcr.astype(int).to_numpy()
 OUT.mkdir(parents=True,exist_ok=True)
 rows=[]; preds=[]
 # Repeated internal CV assesses each acquisition and primary three-way fusion.
 cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED); internal=[]
 for fold,(tr,te) in enumerate(cv.split(E2,y2)):
  P=oof(E2.iloc[tr],M2.iloc[tr],I2.iloc[tr],y2[tr],SEED+fold)
  # Gate is trained only on training-fold OOF expert predictions.
  gate=LogisticRegression(C=1,class_weight='balanced',solver='liblinear').fit(P,y2[tr])
  fitted=experts(E2.iloc[tr],M2.iloc[tr],I2.iloc[tr],y2[tr],SEED+fold)
  Q=np.column_stack([predict_expert(fitted[j],X.iloc[te],k) for j,(X,k) in enumerate([(E2,'rna'),(M2,'clinical'),(I2,'mri')])])
  for name,p in [('rna',Q[:,0]),('clinical',Q[:,1]),('mri',Q[:,2]),('mean3',Q.mean(1)),('fusion3',gate.predict_proba(Q)[:,1]),('no_mri',LogisticRegression(C=1,class_weight='balanced',solver='liblinear').fit(P[:,:2],y2[tr]).predict_proba(Q[:,:2])[:,1])]:
   internal.append({'fold':fold,'model':name,**metrics(y2[te],p)})
 pd.DataFrame(internal).to_csv(OUT/'internal_cv.csv',index=False)
 # External test: all fitting and gate construction use discovery labels only.
 fitted=experts(E2,M2,I2,y2); P=oof(E2,M2,I2,y2); gate=LogisticRegression(C=1,class_weight='balanced',solver='liblinear').fit(P,y2)
 Q=np.column_stack([predict_expert(fitted[j],X,k) for j,(X,k) in enumerate([(E1,'rna'),(M1,'clinical'),(I1,'mri')])])
 no=LogisticRegression(C=1,class_weight='balanced',solver='liblinear').fit(P[:,:2],y2)
 external=[]
 for name,p in [('rna',Q[:,0]),('clinical',Q[:,1]),('mri',Q[:,2]),('mean3',Q.mean(1)),('fusion3',gate.predict_proba(Q)[:,1]),('no_mri',no.predict_proba(Q[:,:2])[:,1])]: external.append({'model':name,'n':len(y1),**metrics(y1,p)})
 pd.DataFrame(external).to_csv(OUT/'external_metrics.csv',index=False)
 pd.DataFrame({'patient_id':M1.index,'pcr':y1,'rna':Q[:,0],'clinical':Q[:,1],'mri':Q[:,2],'fusion3':gate.predict_proba(Q)[:,1],'no_mri':no.predict_proba(Q[:,:2])[:,1]}).to_csv(OUT/'external_predictions.csv',index=False)
 # Patient bootstrap CI; gate-only permutations below are diagnostic, not a pipeline null.
 rng=np.random.default_rng(SEED); delta=[]; primary=gate.predict_proba(Q)[:,1]; ab=no.predict_proba(Q[:,:2])[:,1]
 for _ in range(1000):
  ix=rng.integers(0,len(y1),len(y1))
  if len(np.unique(y1[ix]))==2: delta.append(roc_auc_score(y1[ix],primary[ix])-roc_auc_score(y1[ix],ab[ix]))
 np.save(OUT/'paired_delta_bootstrap.npy',np.array(delta));
 perm=[]
 for b in range(100):
  yp=rng.permutation(y2); g=LogisticRegression(C=1,class_weight='balanced',solver='liblinear').fit(P,yp); perm.append(roc_auc_score(y1,g.predict_proba(Q)[:,1]))
 pd.DataFrame({'permutation':range(len(perm)),'auroc':perm,'observed':roc_auc_score(y1,primary)}).to_csv(OUT/'gate_only_permutations_diagnostic.csv',index=False)
 # Deployment-shift stress test: perturb only the MRI expert probability after all
 # fitting is frozen. This tests whether reliability channels respond to a broken
 # modality rather than treating every source as equally trustworthy.
 rng_shift=np.random.default_rng(SEED+909); stress=[]
 scenarios={'intact':Q[:,2], 'mri_neutralized':np.full(len(Q),.5), 'mri_permuted':rng_shift.permutation(Q[:,2]), 'mri_noisy':np.clip(Q[:,2]+rng_shift.normal(0,.25,len(Q)),.001,.999)}
 for scenario,mri_p in scenarios.items():
  qstress=Q.copy(); qstress[:,2]=mri_p
  stress.extend([{'scenario':scenario,'model':'standard_gate','auroc':roc_auc_score(y1,gate.predict_proba(qstress)[:,1])},{'scenario':scenario,'model':'no_MRI','auroc':roc_auc_score(y1,no.predict_proba(qstress[:,:2])[:,1])}])
 pd.DataFrame(stress).to_csv(OUT/'modality_shift_stress.csv',index=False)
 pd.DataFrame({'cohort':['ISPY2','ISPY1'],'n':[len(y2),len(y1)],'pcr_positive':[int(y2.sum()),int(y1.sum())],'genes':[E2.shape[1],E1.shape[1]],'mri_features':[I2.shape[1],I1.shape[1]]}).to_csv(OUT/'cohort_summary.csv',index=False)
 # Secondary RPPA extension is restricted to I-SPY2 complete cases; no external protein claim.
 rp=pd.read_csv(DATA/'ISPY2_RPPA_expression.csv.gz',index_col=0); rp.index=rp.index.astype(str)
 rid=M2.index.intersection(rp.index); yr=M2.loc[rid,'pcr'].astype(int).to_numpy(); Er=E2.loc[rid]; Mr=M2.loc[rid]; Ir=I2.loc[rid]; Rp=rp.loc[rid].replace([np.inf,-np.inf],np.nan)
 cv4=StratifiedKFold(5,shuffle=True,random_state=SEED+77); extrows=[]
 for fold,(tr,te) in enumerate(cv4.split(Er,yr)):
  three=oof(Er.iloc[tr],Mr.iloc[tr],Ir.iloc[tr],yr[tr],SEED+tr[0]); fourX=np.column_stack([three,Rp.iloc[tr].to_numpy()])
  # OOF expert probabilities plus RPPA features keep the extension explicitly separate.
  four=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,solver='liblinear')).fit(fourX,yr[tr])
  fitted=experts(Er.iloc[tr],Mr.iloc[tr],Ir.iloc[tr],yr[tr],SEED+fold)
  q=np.column_stack([predict_expert(fitted[j],X.iloc[te],k) for j,(X,k) in enumerate([(Er,'rna'),(Mr,'clinical'),(Ir,'mri')])]); q4=np.column_stack([q,Rp.iloc[te].to_numpy()])
  extrows.extend([{'model':'three_acquisition','fold':fold,'auroc':roc_auc_score(yr[te],LogisticRegression(C=1,class_weight='balanced',solver='liblinear').fit(three,yr[tr]).predict_proba(q)[:,1])},{'model':'three_plus_rppa','fold':fold,'auroc':roc_auc_score(yr[te],four.predict_proba(q4)[:,1])}])
 pd.DataFrame(extrows).to_csv(OUT/'rppa_extension_internal.csv',index=False)
 pd.DataFrame({'rppa_complete_case_n':[len(rid)],'rppa_features':[Rp.shape[1]],'external_rppa_validation':[False]}).to_csv(OUT/'rppa_availability.csv',index=False)
 print(pd.DataFrame(external).to_string(index=False)); print('delta bootstrap CI',np.quantile(delta,[.025,.5,.975])); print('RPPA complete cases',len(rid))
if __name__=='__main__': run()
