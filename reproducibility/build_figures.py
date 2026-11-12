"""Read authoritative small results, summarize paired runs and draw the figures remotely."""
from pathlib import Path
import json
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.font_manager import findfont, fontManager

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-root', type=Path, default=ROOT/'inputs')
parser.add_argument('--controlled-root', type=Path, default=ROOT/'reference_results')
parser.add_argument('--output', type=Path, default=ROOT/'generated_figures')
parser.add_argument('--font-file', type=Path, default=None, help='Optional path to an Arial TTF file installed/licensed by the user.')
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
if args.font_file:
    fontManager.addfont(str(args.font_file))
FHN=args.input_root/'fhn'
PK=args.input_root/'pk'
SOURCES=[FHN/'budget_per_run.csv',FHN/'budget_report.json',FHN/'budget_primary.csv',FHN/'budget_secondary.csv',FHN/'budget_method_means.csv',PK/'paired_seed_results.csv']
FONT_FAMILY='Arial'
try:
    FONT=findfont(FONT_FAMILY,fallback_to_default=False)
except ValueError:
    FONT_FAMILY='DejaVu Sans'
    FONT=findfont(FONT_FAMILY,fallback_to_default=False)
plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':7,'axes.linewidth':.65,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.5,'lines.markersize':3.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':450,'axes.unicode_minus':False,'mathtext.fontset':'custom','mathtext.rm':FONT_FAMILY,'mathtext.it':FONT_FAMILY+':italic','mathtext.bf':FONT_FAMILY+':bold'})
BLUE='#0072B2';ORANGE='#D55E00';TEAL='#009E73';GRAY='#757575';PURPLE='#CC79A7'
NOISE_COLORS={.2:BLUE,.6:ORANGE,1.2:TEAL}
plt.rcParams['legend.frameon']=False
GRAPHICS_QA={}

def tick(x,pos=None):
    return '0' if abs(x)<1e-12 else f'{x:g}'

def decorate(ax,letter,title):
    ax.text(-.16,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
    ax.set_title(title,loc='left',pad=9,fontweight='normal')
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    ax.yaxis.set_major_formatter(FuncFormatter(tick))
    ax.tick_params(length=3,width=.65)

def band(ax,x,y,low,high,**kwargs):
    ax.plot(x,y,**kwargs)
    ax.fill_between(x,low,high,color=kwargs.get('color'),alpha=.12,linewidth=0)

def save(fig,name):
    fig.canvas.draw()
    bbox=fig.get_tightbbox(fig.canvas.get_renderer())
    width,height=fig.get_size_inches()
    # Fallback fonts have different text metrics; tight export preserves all labels.
    if FONT_FAMILY != 'Arial':
        margin=0.08
        assert bbox.x0>=-margin and bbox.y0>=-margin and bbox.x1<=width+margin and bbox.y1<=height+margin,(name,bbox.bounds)
    else:
        assert bbox.x0>=0 and bbox.y0>=0 and bbox.x1<=width and bbox.y1<=height,(name,bbox.bounds)
    tick_labels=[label.get_text() for ax in fig.axes for label in ax.get_xticklabels()+ax.get_yticklabels() if label.get_visible()]
    invalid=[s for s in tick_labels if re.fullmatch(r'-?\d+\.\d*0',s) or s=='-0']
    assert not invalid,(name,invalid)
    GRAPHICS_QA[name]={'canvas_inches':[float(width),float(height)],'all_content_within_canvas':True,'content_bbox_inches':[float(x) for x in bbox.bounds],'export_bbox':'tight' if FONT_FAMILY != 'Arial' else 'canvas','tick_labels':tick_labels}
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/f'{name}.{ext}',facecolor='white',**({'bbox_inches':'tight','pad_inches':0.08} if FONT_FAMILY != 'Arial' else {}))
    plt.close(fig)

def bootstrap(values,seed):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return float(values.mean()),*map(float,np.quantile(means,[.025,.975]))

fhn=pd.read_csv(FHN/'budget_per_run.csv',encoding='utf-8-sig')
pk=pd.read_csv(PK/'paired_seed_results.csv',encoding='utf-8-sig')
pk=pk[pk.seed>=2026081205].copy()
assert len(fhn)==300 and fhn.seed.nunique()==50
assert len(pk)==360 and pk.seed.nunique()==45
assert set(fhn.seed).isdisjoint(set(range(2026081200,2026081250)))
report=json.loads((FHN/'budget_report.json').read_text())
fhn_means=[]
metric_names=['nrmse','crps','alpha_absolute_error','coverage_90','interval_width_90','runtime_seconds','measured_core_wall_seconds','forward_member_steps','peak_gpu_memory_mb']
methods=report['setting']['methods']
for mi,method in enumerate(methods):
    frame=fhn[fhn.variant==method].sort_values('seed')
    assert len(frame)==50
    for j,metric in enumerate(metric_names):
        mean,low,high=bootstrap(frame[metric],2026110300+mi*100+j)
        fhn_means.append({'method':method,'metric':metric,'n':50,'mean':mean,'ci_low':low,'ci_high':high})
pd.DataFrame(fhn_means).to_csv(OUT/'fhn_absolute_summary.csv',index=False)
pd.DataFrame(report['primary_comparisons']).to_csv(OUT/'fhn_paired_primary.csv',index=False)
pd.read_csv(FHN/'budget_secondary.csv',encoding='utf-8-sig').to_csv(OUT/'fhn_paired_secondary.csv',index=False)
pk_summary=[]
for mi,method in enumerate(['pce','apce']):
    for factor in [1,2,4,8]:
        group=pk[(pk.method==method)&(pk.observation_factor==factor)].sort_values('seed')
        assert len(group)==45
        for j,metric in enumerate(['primary_error','crps','parameter_error','coverage_90','interval_width_90','runtime_seconds','forward_member_steps']):
            for hi,history in enumerate(['shadow','analysis']):
                mean,low,high=bootstrap(group[f'{history}_{metric}'],2026110400+mi*100+factor*10+j)
                pk_summary.append({'method':method,'factor':factor,'metric':metric,'history':history,'n':45,'mean':mean,'ci_low':low,'ci_high':high})
            mean,low,high=bootstrap(group[f'analysis_{metric}']-group[f'shadow_{metric}'],2026110500+mi*100+factor*10+j)
            pk_summary.append({'method':method,'factor':factor,'metric':metric,'history':'analysis_minus_shadow','n':45,'mean':mean,'ci_low':low,'ci_high':high})
        mean,low,high=bootstrap(group.shadow_mechanism_median_analysis_shadow_ratio,2026110600+mi*100+factor)
        pk_summary.append({'method':method,'factor':factor,'metric':'retained_variance','history':'shadow_run','n':45,'mean':mean,'ci_low':low,'ci_high':high})
pk_summary=pd.DataFrame(pk_summary);pk_summary.to_csv(OUT/'pk_45_summary.csv',index=False)
fhn_summary=pd.DataFrame(fhn_means)
summary=pd.read_csv(args.controlled_root/'controlled_summary.csv')
theory=pd.read_csv(args.controlled_root/'controlled_theory.csv')
config=json.loads((ROOT/'config.json').read_text())

def controlled_figure(group,name):
    data=summary[summary.candidate_set==group]
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.9))
    fig.subplots_adjust(left=.085,right=.985,bottom=.105,top=.93,wspace=.48,hspace=.6)
    ax=axs[0,0];decorate(ax,'a','Candidate forecast separation')
    t=np.arange(1,41)
    for g,color in [(0,BLUE),(.25,ORANGE),(1,TEAL)]:
        h=t if g==0 else (1-(1-g)**t)/g
        ax.plot(t,.04*h,label=f'$g={g:g}$',color=color)
    ax.set(xlabel='Observation index',ylabel='Adjacent mean distance');ax.legend(loc='upper left')
    ax=axs[0,1];decorate(ax,'b','Covariance-weighted separation')
    for sigma,color in NOISE_COLORS.items():
        v=theory[(theory.candidate_set==group)&(theory.noise_sd==sigma)].groupby('feedback').final_nearest_mahalanobis_sq.mean()
        ax.plot(v.index,v.values,color=color,marker='o',label=rf'$\sigma={sigma:g}$')
    ax.set(xlabel='Feedback strength $g$',ylabel='Squared distance / variance',yscale='log');ax.legend()
    ax=axs[0,2];decorate(ax,'c','Selecting the true dynamics')
    for sigma,color in NOISE_COLORS.items():
        d=data[data.noise_sd==sigma].sort_values('feedback')
        band(ax,d.feedback,100*d.correct_mean,100*d.correct_ci_low,100*d.correct_ci_high,color=color,marker='o')
        ax.plot(d.feedback,100*d.exact_correct_mean,color=color,linestyle='--',linewidth=1)
        z=theory[(theory.candidate_set==group)&(theory.noise_sd==sigma)].groupby('feedback').analytic_accuracy.mean()
        ax.plot(z.index,100*z.values,color=color,linewidth=.65,alpha=.6)
    ax.plot([],[],color=GRAY,label='Marginal composite');ax.plot([],[],color=GRAY,linestyle='--',label='Full joint likelihood')
    ax.set(xlabel='Feedback strength $g$',ylabel='Correct selection (%)',ylim=(35,103));ax.legend(loc='lower left',fontsize=6.5)
    d=data[data.noise_sd==.6].sort_values('feedback')
    ax=axs[1,0];decorate(ax,'d',r'State estimates ($\sigma=0.6$)')
    for key,color,label in [('state_uniform_rmse',GRAY,'Uniform candidate mean'),('state_weighted_rmse',BLUE,'Score-weighted mean')]:
        band(ax,d.feedback,d[key+'_mean'],d[key+'_ci_low'],d[key+'_ci_high'],color=color,marker='o',label=label)
    ax.set(xlabel='Feedback strength $g$',ylabel='Mean trajectory RMSE');ax.legend(loc='best',fontsize=6.5)
    ax=axs[1,1];decorate(ax,'e','Forecast after observations stop')
    band(ax,d.feedback,d.blackout_rmse_mean,d.blackout_rmse_ci_low,d.blackout_rmse_ci_high,color=ORANGE,marker='o',label='Marginal composite')
    ax.plot(d.feedback,d.exact_blackout_rmse_mean,color=BLUE,linestyle='--',label='Full joint likelihood')
    ax.set(xlabel='Feedback strength $g$',ylabel='Mean 20-step forecast RMSE');ax.legend(loc='upper left',fontsize=6.5)
    ax=axs[1,2];decorate(ax,'f','Exact probability and simulation')
    for sigma,color in NOISE_COLORS.items():
        d2=data[data.noise_sd==sigma].sort_values('feedback')
        z=theory[(theory.candidate_set==group)&(theory.noise_sd==sigma)].groupby('feedback').analytic_accuracy.mean().sort_index()
        ax.scatter(100*z.values,100*d2.correct_mean,color=color,s=18,edgecolors='white',linewidth=.4)
    ax.plot([35,100],[35,100],color=GRAY,linestyle=':',linewidth=.8)
    ax.set(xlabel='Analytic correct selection (%)',ylabel='Simulated correct selection (%)',xlim=(35,103),ylim=(35,103))
    save(fig,name)

controlled_figure('two','figure1_feedback_control')
controlled_figure('five','extended_feedback_five_candidates')

fig,axs=plt.subplots(3,2,figsize=(7.1,7.1))
fig.subplots_adjust(left=.15,right=.985,bottom=.075,top=.95,wspace=.7,hspace=.65)
ax=axs[0,0];decorate(ax,'a','PK: candidate variance retained')
for method,color,marker in [('pce',BLUE,'o'),('apce',ORANGE,'s')]:
    d=pk_summary[(pk_summary.method==method)&(pk_summary.metric=='retained_variance')].sort_values('factor')
    ax.errorbar(d.factor,100*d['mean'],yerr=[100*(d['mean']-d.ci_low),100*(d.ci_high-d['mean'])],color=color,marker=marker,capsize=2,label=method.upper())
ax.set(xlabel='Observation-interval factor',ylabel='Analysis / isolated variance (%)',xticks=[1,2,4,8]);ax.legend()
ax=axs[0,1];decorate(ax,'b','PK: absolute reconstruction error')
for method,color in [('pce',BLUE),('apce',ORANGE)]:
    for history,ls,marker in [('shadow','-','o'),('analysis','--','s')]:
        d=pk_summary[(pk_summary.method==method)&(pk_summary.history==history)&(pk_summary.metric=='primary_error')].sort_values('factor')
        ax.plot(d.factor,1000*d['mean'],color=color,linestyle=ls,marker=marker,label=f"{method.upper()} {'isolated' if history=='shadow' else 'analysis'}")
ax.set(xlabel='Observation-interval factor',ylabel=r'Mean nRMSE ($\times 10^{-3}$)',xticks=[1,2,4,8]);ax.legend(ncol=2,fontsize=6)
labels={'apce_shadow':'APCE isolated','apce_analysis':'APCE analysis','bma_native':'BMA 30','bma_member_budget':'BMA 45','aug_enkf_native':'Aug-EnKF 30','aug_enkf_member_budget':'Aug-EnKF 310'}
colors={m:BLUE if m=='apce_shadow' else ORANGE if m=='apce_analysis' else TEAL if 'bma' in m else GRAY for m in methods}
ax=axs[1,0];decorate(ax,'c','FHN: absolute reconstruction error')
for i,m in enumerate(methods):
    d=fhn_summary[(fhn_summary.method==m)&(fhn_summary.metric=='nrmse')].iloc[0]
    ax.errorbar(1000*d['mean'],i,xerr=[[1000*(d['mean']-d.ci_low)],[1000*(d.ci_high-d['mean'])]],fmt='o',color=colors[m],capsize=2)
ax.set_yticks(range(6),[labels[m] for m in methods],fontsize=6.5);ax.invert_yaxis();ax.set_xlabel(r'Mean nRMSE ($\times 10^{-3}$)')
ax=axs[1,1];decorate(ax,'d','FHN: paired effect of scoring history')
for i,metric in enumerate(['nrmse','crps']):
    r=next(z for z in report['primary_comparisons'] if z['alternative']=='apce_analysis' and z['metric']==metric)
    ax.errorbar(1000*r['delta_mean'],i,xerr=[[1000*(r['delta_mean']-r['bootstrap_mean_ci95_low'])],[1000*(r['bootstrap_mean_ci95_high']-r['delta_mean'])]],fmt='o',color=BLUE,capsize=3)
    ax.text(.99,.84-i*.43,f"Holm $p={r['holm_adjusted_p']:g}$",transform=ax.transAxes,ha='right',fontsize=7)
ax.axvline(0,color=GRAY,linewidth=.7,linestyle=':');ax.set_yticks([0,1],['nRMSE','CRPS']);ax.set(ylim=(-.55,1.55),xlabel=r'Analysis minus isolated ($\times 10^{-3}$)');ax.invert_yaxis()
ax=axs[2,0];decorate(ax,'e','FHN: measured cost and accuracy')
for i,m in enumerate(methods):
    frame=fhn_summary[fhn_summary.method==m].set_index('metric');x=frame.loc['runtime_seconds','mean'];y=1000*frame.loc['nrmse','mean']
    ax.scatter(x,y,c=colors[m],marker='s' if 'budget' in m else 'o',s=26,edgecolors='white',linewidth=.4)
    ax.annotate(['Isolated','Analysis','BMA 30','BMA 45','Aug 30','Aug 310'][i],(x,y),xytext=([7,7,7,7,7,7][i],[5,4,4,-10,5,-10][i]),textcoords='offset points',fontsize=6.5)
ax.set(xlabel='Measured runtime (s)',ylabel=r'Mean nRMSE ($\times 10^{-3}$)',xlim=(.25,3),ylim=(9.4,12.6))
ax=axs[2,1];decorate(ax,'f','FHN: empirical interval coverage')
for i,m in enumerate(methods):
    d=fhn_summary[(fhn_summary.method==m)&(fhn_summary.metric=='coverage_90')].iloc[0]
    ax.errorbar(100*d['mean'],i,xerr=[[100*(d['mean']-d.ci_low)],[100*(d.ci_high-d['mean'])]],fmt='o',color=colors[m],capsize=2)
ax.axvline(90,color=GRAY,linestyle=':',linewidth=.9);ax.set_yticks(range(6),[labels[m] for m in methods],fontsize=6.5);ax.invert_yaxis();ax.set_xlabel('Empirical 90% interval coverage (%)')
save(fig,'figure2_pk_fhn_controlled')

# Compact additional display retains both primary endpoints, parameter error and measured resources.
fig,axs=plt.subplots(2,3,figsize=(7.1,4.9));fig.subplots_adjust(left=.09,right=.985,bottom=.18,top=.94,wspace=.45,hspace=.55)
for ax,letter,metric,title,scale in zip(axs.flat,'abcdef',['nrmse','crps','alpha_absolute_error','runtime_seconds','forward_member_steps','peak_gpu_memory_mb'],['State error','Distribution error','Parameter error','Measured runtime','Propagation work','Peak GPU memory'],[1000,1000,1,1,1,1]):
    decorate(ax,letter,title)
    for i,m in enumerate(methods):
        d=fhn_summary[(fhn_summary.method==m)&(fhn_summary.metric==metric)].iloc[0]
        ax.errorbar(i,scale*d['mean'],yerr=[[scale*(d['mean']-d.ci_low)],[scale*(d.ci_high-d['mean'])]],fmt='o',color=colors[m],capsize=2)
    ax.set_xticks(range(6),['Iso.','Ana.','B30','B45','A30','A310'],rotation=45,ha='right',fontsize=6.5)
    ax.set_ylabel({'nrmse':r'nRMSE ($\times 10^{-3}$)','crps':r'CRPS ($\times 10^{-3}$)','alpha_absolute_error':'Absolute coordinate error','runtime_seconds':'Time (s)','forward_member_steps':'Member propagation steps','peak_gpu_memory_mb':'Peak allocation (MB)'}[metric])
save(fig,'extended_fhn_accuracy_cost')

for file in [FHN/'budget_report.json',FHN/'budget_method_means.csv',FHN/'budget_primary.csv',FHN/'budget_secondary.csv']:
    (OUT/('source_'+file.name)).write_bytes(file.read_bytes())
registry={'font_family':FONT_FAMILY,'panels':{'figure1_feedback_control':'Two-candidate conditions from config.json, with all generating drifts and 2000 replicates per drift/noise condition.','extended_feedback_five_candidates':'All five-candidate conditions from config.json.','figure2_pk_fhn_controlled_a_b':'PK paired results; first five setting-selection seeds excluded, leaving seeds 2026081205 through 2026081249. Both PCE and APCE retained.','figure2_pk_fhn_controlled_c_d_e_f':'FHN comparison across 50 independent realizations and six methods and ensemble-size settings, with bootstrap intervals for mean outcomes.','extended_fhn_accuracy_cost':'All six FHN methods; includes parameter error, primary errors, propagation cost, time and memory.'},'intervals':'All plotted bootstrap intervals are unadjusted 95% intervals for means across complete paired realizations; paired FHN effects and Holm-adjusted p values use the corresponding paired-realization statistics. Simulated selection intervals are Wilson 95%; simulated mean trajectory errors use normal 95% intervals.'}
(OUT/'figure_sources.json').write_text(json.dumps(registry,indent=2))
qa={'font':FONT_FAMILY,'all_tick_labels_use_general_format':True,'negative_zero_rendered_as_zero':True,'pk_confirmation_count':45,'fhn_confirmation_count':50,'bootstrap_repeats':10000,'controlled_condition_count':len(summary),'outputs':[p.name for p in OUT.glob('*.pdf')]}
qa['graphics_checks']=GRAPHICS_QA
(OUT/'figure_validation.json').write_text(json.dumps(qa,indent=2))
print(json.dumps(qa),flush=True)
