"""Reproduce paired FHN inference from the supplied 300 per-run rows.

The bootstrap, sign-flip seeds, comparison order and Holm family follow
the original assess_adoption.py in the data source record.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,default=ROOT/'inputs/fhn/budget_per_run.csv')
parser.add_argument('--output',type=Path,default=ROOT/'generated_statistics')
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
frame=pd.read_csv(args.input,encoding='utf-8-sig',float_precision='round_trip')
assert len(frame)==300 and frame.seed.nunique()==50
shadow=frame[frame.variant=='apce_shadow'].set_index('seed').sort_index()
primary,secondary=[],[]
metrics=['nrmse','crps','alpha_absolute_error','coverage_90','coverage_error','interval_width_90','runtime_seconds','forward_member_steps','peak_gpu_memory_mb']
for index,label in enumerate(sorted(set(frame.variant)-{'apce_shadow'})):
    alt=frame[frame.variant==label].set_index('seed').sort_index()
    assert alt.index.equals(shadow.index)
    for j,metric in enumerate(metrics):
        a=alt[metric].to_numpy();s=shadow[metric].to_numpy();delta=a-s
        rng=np.random.default_rng(20261909+index*100+j)
        means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
        target=abs(delta.mean());threshold=max(target-1e-12*max(target,1e-30),0.)
        signs=2*rng.integers(0,2,(19999,len(delta)))-1
        p=float((1+np.sum(np.abs((signs*delta).mean(axis=1))>=threshold))/20000)
        row={'n':len(delta),'delta_mean':float(delta.mean()),'bootstrap_mean_ci95_low':float(np.quantile(means,.025)),'bootstrap_mean_ci95_high':float(np.quantile(means,.975)),'positive_count':int(np.sum(delta>0)),'negative_count':int(np.sum(delta<0)),'two_sided_sign_flip_p':p,'metric':metric,'alternative_mean':float(a.mean()),'shadow_mean':float(s.mean()),'relative_reduction_ratio_of_means_pct':100*float(a.mean()-s.mean())/max(abs(float(a.mean())),1e-15),'mean_paired_relative_reduction_pct':100*float(np.mean(delta/np.maximum(np.abs(a),1e-15))),'alternative':label}
        (primary if j<2 else secondary).append(row)
previous=0.
for rank,row in enumerate(sorted(primary,key=lambda r:r['two_sided_sign_flip_p'])):
    previous=min(1.,max(previous,(len(primary)-rank)*row['two_sided_sign_flip_p']))
    row.update(holm_adjusted_p=previous,holm_family_size=len(primary),direction_supported_after_holm=previous<.05)
pd.DataFrame(primary).to_csv(args.output/'fhn_paired_primary.csv',index=False)
pd.DataFrame(secondary).to_csv(args.output/'fhn_paired_secondary.csv',index=False)
print('Recomputed 10 primary and 35 secondary paired FHN comparisons.')
