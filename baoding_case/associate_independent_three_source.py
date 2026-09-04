#!/usr/bin/env python3
"""Calibrate then associate independent raw-acoustic three-source peaks.

GPS is deliberately confined to a short initial calibration interval.  After
that interval target labels are propagated using node-angle continuity and a
robust multi-ray position prediction, never GPS.  The output is a target-wise
PCE/APCE observation interface with quality and provenance fields.
"""
from __future__ import annotations

import argparse, csv, itertools, json, math, statistics
from pathlib import Path

import shuangyuan_dual_association as geom

NODES=(1,3,5,6,7,8,11,13)
TARGETS=(1,2,3)
GPS_FILES={1:'GPS1_plane1.gps',2:'GPS3_plane2.gps',3:'GPS4_plane2to3.gps'}
PERMS=tuple(itertools.permutations(range(3)))

def read_csv(path:Path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def parse_gps(path:Path):
    rows=[]
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        p=line.split()
        if len(p)>=8:
            try: rows.append((geom.hms_seconds(float(p[7])),(float(p[4]),float(p[5]),float(p[6]))))
            except ValueError:pass
    return rows

def near(track,t): return min(track,key=lambda x:abs(x[0]-t))[1]
def circ(a,b): return (a-b+180)%360-180
def angles(pos,node): return geom.truth_angles(pos,node)
def raw(row): return ([float(row[f'azimuth_{i}_deg']) for i in TARGETS],[float(row[f'zenith_{i}_deg']) for i in TARGETS])
def transform(a,z,c): return ([(c['az_sign']*x+c['az_offset_deg'])%360 for x in a],[c['zen_sign']*x+c['zen_offset_deg'] for x in z])

def frame_cost(a,z,truth,az_perm,ze_perm):
    return sum(math.hypot(circ(a[az_perm[k]],truth[k][0]),z[ze_perm[k]]-truth[k][1]) for k in range(3))

def calibrate(rows,node,gps,indices):
    best=None
    # The expensive offset grid is unnecessary: for each orientation pair,
    # estimate circular/linear offsets from the best assignment and iterate.
    for sign in (1.,-1.):
      for zsign in (1.,-1.):
        azoff=0.; zoff=0.; matched=[]
        for _ in range(3):
          matched=[]; az_res=[]; z_res=[]; total=0.
          for i in indices:
            ra,rz=raw(rows[i]); a=[(sign*x+azoff)%360 for x in ra]; z=[zsign*x+zoff for x in rz]
            truth=[angles(near(gps[t],float(rows[i]['time_s'])),node) for t in TARGETS]
            cand=min((frame_cost(a,z,truth,ap,zp),ap,zp) for ap in PERMS for zp in PERMS)
            total+=cand[0]; matched.append(cand)
            for k in range(3):
              az_res.append(circ(truth[k][0], sign*ra[cand[1][k]]))
              z_res.append(truth[k][1]-zsign*rz[cand[2][k]])
          if az_res: azoff=math.degrees(math.atan2(sum(math.sin(math.radians(x)) for x in az_res),sum(math.cos(math.radians(x)) for x in az_res)))%360
          if z_res: zoff=statistics.median(z_res)
        candidate=(total,sign,azoff,zsign,zoff)
        if best is None or candidate[0]<best[0]: best=candidate
    total,sign,azoff,zsign,zoff=best
    return {'az_sign':sign,'az_offset_deg':float(azoff),'zen_sign':zsign,'zen_offset_deg':float(zoff),'calibration_objective_deg':total/max(len(indices),1),'calibration_frames':len(indices)}

def choose(a,z,cal,prev,node):
    a,z=transform(a,z,cal)
    if prev is None:
        return [(a[k],z[k]) for k in range(3)],(0,1,2),(0,1,2),0.
    # Pair azimuth and zenith permutations jointly against the predicted
    # three-target geometry; this is independent of GPS after calibration.
    truth=[angles(p,node) if p is not None else (0.,45.) for p in prev]
    best=min((frame_cost(a,z,truth,ap,zp),ap,zp) for ap in PERMS for zp in PERMS)
    return [(a[best[1][k]],z[best[2][k]]) for k in range(3)],best[1],best[2],best[0]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--gps-root',type=Path,required=True);ap.add_argument('--nod',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--calibration-start',type=int,default=132754);ap.add_argument('--calibration-seconds',type=float,default=10.);args=ap.parse_args()
 rows={n:read_csv(args.input_root/f'node{n}'/f'node{n}_independent_observations.csv') for n in NODES}; count=min(map(len,rows.values()));rows={n:v[:count] for n,v in rows.items()}
 node_xyz=geom.parse_nod(args.nod); center=tuple(sum(node_xyz[n][d] for n in NODES)/len(NODES) for d in range(3)); gps={t:parse_gps(args.gps_root/f) for t,f in GPS_FILES.items()}; times=[float(rows[NODES[0]][i]['time_s']) for i in range(count)]; start=min(range(count),key=lambda i:abs(times[i]-geom.hms_seconds(args.calibration_start))); stop=min(count,start+max(1,int(args.calibration_seconds/(times[1]-times[0])))); indices=list(range(start,stop))
 calibrations={n:calibrate(rows[n],node_xyz[n],gps,indices) for n in NODES}
 out={t:[] for t in TARGETS}; prev=None; accepted=0
 for i in range(start,count):
  paired={}; costs=[]
  for n in NODES:
   a,z=raw(rows[n][i]); p,_,_,cost=choose(a,z,calibrations[n],prev,node_xyz[n]); paired[n]=p;costs.append(cost)
  pos=[]; inliers=[]
  for k in range(3):
   point,ins,cond=geom.robust_triangulate({n:paired[n][k] for n in NODES},node_xyz);pos.append(point);inliers.append((ins,cond))
  if all(x is not None for x in pos):prev=pos;accepted+=1
  for k,t in enumerate(TARGETS):
   for n in NODES:
    az,ze=paired[n][k];q=float(rows[n][i]['quality_score']);out[t].append({'time_s':times[i],'frame_index':i,'target_id':t,'node_id':n,'azimuth_deg':az,'elevation_deg':90-ze,'concentration':max(0.01,q),'valid':pos[k] is not None and len(inliers[k][0])>=5,'quality_score':q,'association_cost_deg':costs[NODES.index(n)],'inlier_nodes':len(inliers[k][0]),'condition_number':inliers[k][1],'observation_source':'independent_raw_wavfm_music'})
 args.output_root.mkdir(parents=True,exist_ok=True)
 for t,vals in out.items():
  d=args.output_root/f'target{t}';d.mkdir(exist_ok=True)
  with (d/'observations.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(vals[0]));w.writeheader();w.writerows(vals)
  truth=[]
  for i in range(start,count):
   p=near(gps[t],times[i]);truth.append({'time_s':times[i],'px':p[0]-center[0],'py':p[1]-center[1],'pz':p[2]-center[2]})
  with (d/'gps_truth.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(truth[0]));w.writeheader();w.writerows(truth)
  fm={'nodes':{str(n):{'x':node_xyz[n][0]-center[0],'y':node_xyz[n][1]-center[1],'z':node_xyz[n][2]-center[2]} for n in NODES},'paper_nodes':list(NODES),'coordinate_frame':'centered local ENU, metres','input_provenance':{'independent_acoustic_observation':True,'synthetic_observation_from_dbn':False,'source':'raw wavfm → wideband 3-peak MUSIC → calibration/association','gps_runtime_observation':False,'gps_role':'short initial angle orientation/target-order calibration and post-hoc evaluation only'}}
  (d/'frontend_manifest.json').write_text(json.dumps(fm,ensure_ascii=False,indent=2),encoding='utf-8');(d/'frontend_calibration.json').write_text(json.dumps({'calibration':calibrations,'calibration_start_hhmmss':args.calibration_start,'calibration_seconds':args.calibration_seconds},ensure_ascii=False,indent=2),encoding='utf-8')
 man={'claim_status':'independent_three_source_association','nodes':list(NODES),'frames_input':count,'frames_after_calibration':count-start,'accepted_all_target_fraction':accepted/max(count-start,1),'gps_used_after_calibration':False,'calibration':calibrations,'target_mapping':GPS_FILES,'warning':'Association result requires quantitative admission before PCE/APCE results can be promoted beyond diagnostic status.'};(args.output_root/'association_manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(man,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
