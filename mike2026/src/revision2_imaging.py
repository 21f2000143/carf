"""Compact, outcome-independent features from pretreatment DCE-MRI volumes.

The input arrays have (slice, row, column) order. Their export affine is identity;
physical spacing therefore comes from the dataset's accompanying metadata.
These are explicitly defined image features, not an IBSI/PyRadiomics claim.
"""
import numpy as np
from scipy.stats import skew, kurtosis


def summaries(values, prefix):
    values=np.asarray(values,dtype=float)
    quantiles=np.quantile(values,[.1,.25,.5,.75,.9])
    spread=float(values.std())
    out={f'{prefix}_mean':float(values.mean()),f'{prefix}_std':spread}
    out.update({f'{prefix}_q{q}':float(v) for q,v in zip([10,25,50,75,90],quantiles)})
    out[f'{prefix}_skew']=float(skew(values)) if spread>1e-10 else 0.
    out[f'{prefix}_kurtosis']=float(kurtosis(values)) if spread>1e-10 else 0.
    return out


def image_features(pre, early, late, mask, spacing):
    arrays=[np.asarray(x,dtype=float) for x in [pre,early,late]]
    pre,early,late=arrays
    mask=np.asarray(mask)>0
    if mask.ndim!=3 or any(x.shape!=mask.shape for x in arrays):
        raise ValueError('Image and mask dimensions do not agree')
    if mask.sum()<10: raise ValueError('Fewer than ten tumour-mask voxels')
    spacing=np.asarray(spacing,dtype=float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('Invalid physical voxel spacing')
    if not all(np.isfinite(x[mask]).all() for x in arrays):
        raise ValueError('Nonfinite image signal inside mask')
    baseline=pre[mask]
    positive=baseline[baseline>0]
    if len(positive)==0: raise ValueError('No positive baseline tumour signal')
    scale=float(np.median(positive))
    denominator=np.maximum(np.abs(pre),max(float(np.quantile(positive,.95))*.01,1e-12))
    enhancement=(early-pre)/denominator
    late_enhancement=(late-pre)/denominator
    change=(late-early)/denominator
    out={}
    for name,values in [('pre',pre[mask]/scale),('early',early[mask]/scale),
                        ('late',late[mask]/scale),('enhancement',np.clip(enhancement[mask],-5,20)),
                        ('late_enhancement',np.clip(late_enhancement[mask],-5,20)),
                        ('late_change',np.clip(change[mask],-5,20))]:
        out.update(summaries(values,name))
    coords=np.argwhere(mask)
    lengths=(coords.max(axis=0)-coords.min(axis=0)+1)*spacing
    volume=float(mask.sum()*np.prod(spacing))
    padded=np.pad(mask.astype(np.int8),1)
    surface=sum(float(np.abs(np.diff(padded,axis=axis)).sum()*np.prod(np.delete(spacing,axis)))
                for axis in range(3))
    out.update(log_volume_mm3=float(np.log1p(volume)),log_surface_mm2=float(np.log1p(surface)),
               box_extent=float(volume/np.prod(lengths)),
               sphericity=float(np.pi**(1/3)*(6*volume)**(2/3)/surface),
               axis_ratio=float(lengths.min()/lengths.max()),
               log_max_extent_mm=float(np.log1p(lengths.max())),
               washout_fraction=float(np.mean((enhancement[mask]>.1)&(change[mask]<-.1))),
               persistent_fraction=float(np.mean((enhancement[mask]>.1)&(change[mask]>.1))))
    # Fixed-bin, masked nearest-neighbour enhancement texture over three axes.
    bins=np.clip(((np.clip(enhancement,-1,5)+1)/6*16).astype(int),0,15)
    co=np.zeros((16,16),dtype=float)
    for axis in range(3):
        a=[slice(None)]*3; b=a.copy(); a[axis]=slice(None,-1); b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b); valid=mask[a]&mask[b]
        np.add.at(co,(bins[a][valid],bins[b][valid]),1)
    co=co+co.T
    if co.sum()==0: raise ValueError('No adjacent tumour-mask voxels')
    co/=co.sum(); i,j=np.indices(co.shape)
    out['texture_contrast']=float(np.sum(co*(i-j)**2))
    out['texture_homogeneity']=float(np.sum(co/(1+(i-j)**2)))
    out['texture_energy']=float(np.sum(co**2))
    out['texture_entropy']=float(-np.sum(co[co>0]*np.log2(co[co>0])))
    if not np.isfinite(list(out.values())).all(): raise ValueError('Nonfinite image feature')
    return out
