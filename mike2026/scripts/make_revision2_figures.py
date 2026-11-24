import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/revision2'; F=ROOT/'paper/figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'figure.dpi':150,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
# Architecture: data provenance and leakage-safe fusion.
fig,ax=plt.subplots(figsize=(10,4)); ax.axis('off')
boxes=[(0.02,.62,.22,.22,'Pretreatment\nDCE-MRI\n3 phases + mask'),(.02,.30,.22,.22,'Microarray\ntranscriptomics\ncommon-gene ranks'),(.02,.0,.22,.22,'Clinical\nage, HR, HER2'),(.40,.62,.20,.22,'MRI expert\n66 features'),(.40,.30,.20,.22,'RNA expert\n3,000 genes'),(.40,.0,.20,.22,'Clinical expert'),(.75,.30,.20,.28,'Logistic gate\non expert\nprobabilities')]
for x,y,w,h,t in boxes: ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',fc='#e8f1fb' if x<.3 else '#f8efe0',ec='#35618c')); ax.text(x+w/2,y+h/2,t,ha='center',va='center')
for a,b in [((.24,.73),(.40,.73)),((.24,.41),(.40,.41)),((.24,.11),(.40,.11)),((.60,.73),(.75,.46)),((.60,.41),(.75,.46)),((.60,.11),(.75,.46))]: ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=12,lw=1.3,color='#333'))
ax.text(.76,.16,'External pCR\nprobability',ha='center',va='center',fontsize=10,weight='bold'); fig.savefig(F/'architecture.pdf'); plt.close(fig)
# Flow.
fig,ax=plt.subplots(figsize=(8,4)); ax.axis('off'); steps=[('I-SPY2 public records','985 clinical\n988 expression\n982 image records'),('Matched acquisitions','978 candidates\n2 MRI QC exclusions'),('Discovery analysis','976 complete cases\n315 pCR'),('I-SPY1 external test','112 complete cases\n31 pCR')]
for i,(a,b) in enumerate(steps):
 x=.02+i*.245; ax.add_patch(FancyBboxPatch((x,.28),.19,.42,boxstyle='round,pad=.02',fc=['#e8f1fb','#e8f1fb','#f8efe0','#e8f1fb'][i],ec='#35618c')); ax.text(x+.095,.59,a,ha='center',va='center',weight='bold'); ax.text(x+.095,.43,b,ha='center',va='center');
 if i<3: ax.add_patch(FancyArrowPatch((x+.19,.49),(x+.245,.49),arrowstyle='->',mutation_scale=12))
fig.savefig(F/'cohort_flow.pdf'); plt.close(fig)
# External bar plot.
ext=pd.read_csv(R/'external_metrics.csv'); fig,ax=plt.subplots(figsize=(7,4)); order=['clinical','rna','mri','mean3','fusion3','no_mri']; d=ext.set_index('model').loc[order]; bars=ax.bar([x.replace('_',' ') for x in order],d.auroc,color=['#6b8e23','#888','#b56576','#d9a441','#35618c','#222']); ax.set_ylim(0,1); ax.set_ylabel('External AUROC (I-SPY1)'); ax.tick_params(axis='x',rotation=35); ax.axhline(.5,color='#999',lw=.8); ax.bar_label(bars,fmt='%.3f',padding=2,fontsize=8); fig.tight_layout(); fig.savefig(F/'external_auroc.pdf'); plt.close(fig)
# Internal repeated CV distribution.
cv=pd.read_csv(R/'internal_cv.csv'); order=['clinical','rna','mri','mean3','fusion3','no_mri']; fig,ax=plt.subplots(figsize=(7,4)); vals=[cv.loc[cv.model==x,'auroc'] for x in order]; ax.boxplot(vals,labels=[x.replace('_',' ') for x in order],showmeans=True); ax.set_ylim(0,1); ax.set_ylabel('AUROC across 25 internal folds'); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(F/'internal_cv.pdf'); plt.close(fig)
# Reliability diagram external.
pd_=pd.read_csv(R/'external_predictions.csv'); fig,ax=plt.subplots(figsize=(5,4)); bins=np.linspace(0,1,6)
for col,c in [('fusion3','#35618c'),('no_mri','#222'),('clinical','#6b8e23')]:
 x=[]; y=[]
 for lo,hi in zip(bins[:-1],bins[1:]):
  q=pd_[(pd_[col]>=lo)&(pd_[col]<hi)]
  if len(q): x.append(q[col].mean()); y.append(q.pcr.mean())
 ax.plot(x,y,'o-',label=col.replace('_',' '),color=c)
ax.plot([0,1],[0,1],'--',color='#999'); ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed pCR fraction'); ax.legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(F/'reliability.pdf'); plt.close(fig)
# RPPA extension.
rp=pd.read_csv(R/'rppa_extension_internal.csv'); g=rp.groupby('model').auroc.agg(['mean','std']).reindex(['three_acquisition','three_plus_rppa']); fig,ax=plt.subplots(figsize=(5,4)); bars=ax.bar(['MRI + RNA + clinical','+ RPPA'],g['mean'],yerr=g['std'],capsize=4,color=['#35618c','#b56576']); ax.set_ylim(0,1); ax.set_ylabel('I-SPY2 internal AUROC'); ax.bar_label(bars,fmt='%.3f',padding=2); fig.tight_layout(); fig.savefig(F/'rppa_extension.pdf'); plt.close(fig)

# Modality-shift stress test.
st=pd.read_csv(R/'modality_shift_stress.csv'); piv=st.pivot(index='scenario',columns='model',values='auroc').reindex(['intact','mri_neutralized','mri_noisy','mri_permuted']); fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(piv)); w=.35
for j,col in enumerate(['no_MRI','standard_gate']): ax.bar(x+(j-.5)*w,piv[col],w,label=col.replace('_',' '))
ax.set_xticks(x,['intact','MRI neutralized','MRI noisy','MRI permuted']); ax.set_ylim(0,1); ax.set_ylabel('External AUROC'); ax.legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(F/'modality_shift_stress.pdf'); plt.close(fig)
