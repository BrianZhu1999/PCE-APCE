#!/usr/bin/env python3
"""Independent eight-node three-source acoustic observation frontend.

This renderer uses the raw ``.wavfm`` streams only.  It deliberately does not
consume DBN states.  Three peaks are extracted from the wideband spatial
spectrum and exported with quality fields; target association is a separate
calibration/audit stage.
"""
from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import shuangyuan_dual_frontend as base

PAPER_NODES=(1,3,5,6,7,8,11,13)
NODE_TO_IP={1:47,3:40,5:54,6:43,7:49,8:61,11:5,13:46}
FS=3050; NFFT=2048; SNAPSHOT=2048; STEP=1024; K=3; C=340.0; SPACING=0.5

def freq_grid():
    fft_f=np.arange(NFFT//2+1)*FS/NFFT
    wanted=np.arange(3,1498,3,dtype=float)
    bins=np.asarray([int(np.argmin(np.abs(fft_f-f))) for f in wanted],dtype=int)
    return bins, fft_f[bins], wanted

def decompose(block,bins):
    frames=block[:,:SNAPSHOT].reshape(block.shape[0],1,SNAPSHOT)
    spectrum=np.fft.fft(frames,axis=-1)/math.sqrt(SNAPSHOT)
    return spectrum[:,:,bins]

def score(coeff, freqs, angles, axis):
    if axis=='azimuth':
        deploy=np.asarray([-3,-2,-1,1,2,3],float)*SPACING
        deployment=np.concatenate([deploy*np.cos(np.deg2rad(angles))[:,None], -deploy*np.sin(np.deg2rad(angles))[:,None]],axis=1)
    else:
        deploy=np.asarray([-3,-2,-1,0,1,2,3],float)*SPACING
        deployment=deploy[None,:]*np.cos(np.deg2rad(angles))[:,None]
    total=np.zeros(len(angles),float)
    for fi,f in enumerate(freqs):
        xyz=coeff[:,:,fi]
        cov=xyz@xyz.conj().T/max(coeff.shape[1],1)
        _,vec=np.linalg.eigh(cov); noise=vec[:,:max(1,vec.shape[1]-K)]
        steering=np.exp(1j*2*np.pi*float(f)/C*deployment.T)
        proj=steering.conj().T@noise; total += np.sum(np.abs(proj)**2,axis=1).real
    return 1/np.maximum(total,1e-12)

def peaks(coeff,freqs,axis):
    low,high=(0.,360.) if axis=='azimuth' else (0.,90.)
    coarse=np.arange(low,high+1e-9,3.0)
    s=score(coeff,freqs,coarse,axis); selected=[]
    step=3.0
    for i in np.argsort(s)[::-1]:
        a=coarse[int(i)]; ds=[min(abs(a-x),360-abs(a-x)) if axis=='azimuth' else abs(a-x) for x in selected]
        if all(d>=12 for d in ds): selected.append(a)
        if len(selected)==K: break
    out=[]; strength=[]
    for a in selected:
        fine=np.clip(np.arange(a-3,a+3.01,1.),low,high); sf=score(coeff,freqs,fine,axis); j=int(np.argmax(sf)); out.append(float(fine[j])); strength.append(float(sf[j]))
    order=np.argsort(strength)[::-1]; return np.asarray(out)[order], np.asarray(strength)[order]

def extract(data,bins,freqs):
    z,x,y=base.channel_vectors(data[:19,:SNAPSHOT]); xc=decompose(x,bins); yc=decompose(y,bins); zc=decompose(z,bins)
    az,asz=peaks(np.concatenate([xc,yc],axis=0),freqs,'azimuth'); ze,zsz=peaks(zc,freqs,'elevation')
    row={}
    for i in range(K): row.update({f'azimuth_{i+1}_deg':float(az[i]),f'zenith_{i+1}_deg':float(ze[i]),f'azimuth_strength_{i+1}':float(asz[i]),f'zenith_strength_{i+1}':float(zsz[i])})
    row['quality_score']=float(np.mean(np.log1p(np.asarray(asz)+np.asarray(zsz))))
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--remote-root',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); ap.add_argument('--node',type=int,required=True); ap.add_argument('--max-frames',type=int); args=ap.parse_args()
    seg=args.remote_root/'20171107保定实验/project/20171107baoding/sanyuan_tongxinyuan_6'; ip=NODE_TO_IP[args.node]; prefix=seg/f'20171107baoding_132614_{ip}_19'; data,meta=base.decode_wavfm(prefix.with_suffix('.wavfm')); bins,freqs,wanted=freq_grid(); n=max(0,1+(data.shape[1]-SNAPSHOT)//STEP); n=min(n,args.max_frames) if args.max_frames else n
    rows=[]
    for i in range(n):
        st=i*STEP; tic=time.monotonic(); r=extract(data[:,st:st+SNAPSHOT],bins,freqs); r.update({'node_id':args.node,'ip_suffix':ip,'frame_index':i,'time_s':base.hms_seconds(132614)+st/FS,'frame_start_sample':st,'frontend_runtime_s':time.monotonic()-tic}); rows.append(r)
    args.output_root.mkdir(parents=True,exist_ok=True); out=args.output_root/f'node{args.node}_independent_observations.csv'
    with out.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []); w.writeheader(); w.writerows(rows)
    manifest={'claim_status':'independent_acoustic_observation_frontend','node':args.node,'ip_suffix':ip,'raw_wavfm':meta,'sample_rate_hz':FS,'snapshot_length_samples':SNAPSHOT,'snapshot_duration_s':SNAPSHOT/FS,'step_samples':STEP,'overlap_fraction':1-STEP/SNAPSHOT,'frequency_grid_hz':wanted.tolist(),'frequency_count':len(wanted),'sources':K,'observation_source':'raw_wavfm_wideband_music','dbn_input_used':False,'target_labels':'not_assigned_until_separate_association','rows':len(rows),'csv':str(out)}
    (args.output_root/f'node{args.node}_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'node':args.node,'rows':len(rows),'runtime_s':sum(float(r['frontend_runtime_s']) for r in rows)},ensure_ascii=False))
if __name__=='__main__': main()
