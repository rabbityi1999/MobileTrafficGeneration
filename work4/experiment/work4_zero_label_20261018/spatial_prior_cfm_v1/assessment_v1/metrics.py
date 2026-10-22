# -*- coding: utf-8 -*-
import numpy as np

def score(y, samples):
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = s.shape[1]
    hp = (np.sort(s, axis=1) * (2*np.arange(1,m+1)-m-1)[None,:,None]).sum(1)/m**2
    per = (np.abs(s-y[:,None]).mean(1)-hp).mean(1)
    lo, hi = np.quantile(s, [.05,.95], axis=1)
    covered = (lo <= y) & (y <= hi)
    result = {'sample_crps':float(per.mean()), 'fair_crps':float(per.mean()-hp.mean()/(m-1)),
              'coverage90':float(covered.mean()), 'width90':float((hi-lo).mean()),
              'sample_mean_mae':float(np.abs(s.mean(1)-y).mean())}
    for name, mask in [('zero',y<=1e-7),('one',y>=1-1e-7),('interior',(y>1e-7)&(y<1-1e-7))]:
        result['coverage90_'+name] = float(covered[mask].mean()) if mask.any() else None
        result['count_'+name] = int(mask.sum())
    # Fixed lag variogram components: squared error of expected absolute increment.
    # Not a strict multivariate score or a lag-168 autocorrelation estimate.
    for lag in [1,24,48]:
        pred = np.abs(s[:,:,lag:]-s[:,:,:-lag]).mean(1)
        truth = np.abs(y[:,lag:]-y[:,:-lag])
        result['variogram_lag%d'%lag] = float(((pred-truth)**2).mean())
    result['variogram_mean'] = float(np.mean([result['variogram_lag%d'%lag] for lag in [1,24,48]]))
    return result, per
