"""Run the scalar-feedback simulation and save numerical summaries."""
from pathlib import Path
import csv
import argparse
import json
import time
import numpy as np
from scipy.special import ndtr, softmax

ROOT = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=ROOT/'config.json')
parser.add_argument('--output', type=Path, default=ROOT/'generated_controlled')
args=parser.parse_args()
C = json.loads(args.config.read_text())
OUT = args.output
OUT.mkdir(exist_ok=False)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def wilson(k, n):
    p, z = k / n, 1.959963984540054
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return float(mid-half), float(mid+half)


start = time.perf_counter()
T, H, N = C["observation_count"], C["blackout_steps"], C["replicates_per_truth_noise"]
t = np.arange(1, T+1, dtype=float)
horizon = np.arange(1, H+1, dtype=float)
all_rows, theory_rows, history_rows, tests = [], [], [], []
full_cov_max_error = 0.0
# The two-observation limit checks the distinct marginal and joint score geometries.
M2=np.array([[1.,0.],[-1.,1.]])
h2=M2@np.array([1.,2.]);V2=M2@M2.T;v2=h2/np.diag(V2)
assert np.isclose(v2@V2@v2/(h2@v2)**2,2/9)
assert np.isclose(np.linalg.det(V2),1)
for group_index, (name, candidate_list) in enumerate(C["candidate_sets"].items()):
    b = np.asarray(candidate_list, dtype=float)
    for true_index, truth in enumerate(b):
        rng = np.random.default_rng(np.random.SeedSequence([C["seed"], group_index, true_index]))
        standard_noise = rng.normal(size=(N, T))
        x = truth * t
        for sigma in C["observation_noise_sd"]:
            noise = sigma*standard_noise
            y = x + noise
            exact_residual = y[:, None, :] - b[None, :, None]*t
            exact_log = -np.sum(exact_residual**2, axis=2)/(2*sigma*sigma)
            exact_selected = np.argmax(exact_log, axis=1)
            exact_weight = softmax(exact_log, axis=1)
            exact_drift = exact_weight @ b
            for g in C["feedback_strengths"]:
                q = 1-g
                # Invertible residual transform: diagonal entries are one.
                M = np.eye(T)
                for i in range(T):
                    for j in range(i):
                        M[i,j] = -g*q**(i-j-1)
                shape = M @ t
                cov = M @ M.T
                marginal_variance = sigma*sigma*np.diag(cov)
                transformed = np.einsum("ij,nkj->nki", M, exact_residual)
                score = -np.sum(transformed**2/marginal_variance, axis=2)/2
                # Full covariance must recover the original joint likelihood.
                test_res = transformed[:3].reshape(-1, T)
                back = np.linalg.solve(M, test_res.T).T
                check = -np.sum(back**2, axis=1)/(2*sigma*sigma)
                error = float(np.max(np.abs(check-exact_log[:3].reshape(-1))))
                full_cov_max_error = max(full_cov_max_error, error)
                assert error < 1e-7
                if g == 0:
                    assert np.allclose(score, exact_log, atol=1e-9, rtol=1e-12)
                selected = np.argmax(score, axis=1)
                weights = softmax(score, axis=1)
                weighted_drift = weights @ b
                pred = np.zeros((N, len(b)))
                cumulative = np.zeros_like(pred)
                uniform_sq, weighted_sq = np.zeros(N), np.zeros(N)
                for it in range(T):
                    pred += b
                    residual = y[:, it, None]-pred
                    assert np.allclose(residual, transformed[:, :, it], atol=1e-10)
                    cumulative -= residual**2/(2*marginal_variance[it])
                    pred += g*residual
                    uniform_estimate = pred.mean(axis=1)
                    weighted_estimate = np.sum(softmax(cumulative, axis=1)*pred, axis=1)
                    uniform_sq += (uniform_estimate-x[it])**2
                    weighted_sq += (weighted_estimate-x[it])**2
                    if name == "two" and truth == 0.02 and sigma == 0.6 and g in (0, 1):
                        history_rows.append({"feedback":g,"time":it+1,"truth_state":x[it],"observation":y[0,it],"candidate_low_forecast":y[0,it]-residual[0,0],"candidate_high_forecast":y[0,it]-residual[0,1],"uniform_analysis":uniform_estimate[0],"weighted_analysis":weighted_estimate[0]})
                assert np.allclose(cumulative, score, atol=1e-8)
                wrong = score.copy()
                wrong[:,true_index] = -np.inf
                margin = score[:,true_index]-wrong.max(axis=1)
                base_error = y[:,-1]-truth*T
                forecast_error = base_error[:,None]+(b[selected]-truth)[:,None]*horizon
                exact_forecast_error = base_error[:,None]+(b[exact_selected]-truth)[:,None]*horizon
                weighted_forecast_error = base_error[:,None]+(weighted_drift-truth)[:,None]*horizon
                exact_weighted_error = base_error[:,None]+(exact_drift-truth)[:,None]*horizon
                arrays = dict(correct=(selected==true_index).astype(int),exact_correct=(exact_selected==true_index).astype(int),true_score_margin=margin,state_uniform_rmse=np.sqrt(uniform_sq/T),state_weighted_rmse=np.sqrt(weighted_sq/T),blackout_rmse=np.sqrt(np.mean(forecast_error**2,axis=1)),exact_blackout_rmse=np.sqrt(np.mean(exact_forecast_error**2,axis=1)),weighted_blackout_rmse=np.sqrt(np.mean(weighted_forecast_error**2,axis=1)),exact_weighted_blackout_rmse=np.sqrt(np.mean(exact_weighted_error**2,axis=1)),drift_horizon_error=H*np.abs(b[selected]-truth),selected_drift=b[selected],weighted_drift=weighted_drift)
                for rep in range(N):
                    row={"candidate_set":name,"truth":float(truth),"noise_sd":sigma,"feedback":g,"replicate":rep}
                    row.update({key:float(value[rep]) for key,value in arrays.items()})
                    all_rows.append(row)
                nearest_delta = float(np.min(np.abs(b[np.arange(len(b))!=true_index]-truth)))
                v = shape/np.diag(cov)
                expected_margin=nearest_delta**2*float(shape@v)/(2*sigma*sigma)
                z=abs(nearest_delta)*float(shape@v)/(2*sigma*np.sqrt(v@cov@v))
                drift_sd=sigma*np.sqrt(v@cov@v)/(shape@v)
                exact_sd=sigma/np.linalg.norm(t)
                left=(b[true_index-1]+truth)/2 if true_index else -np.inf
                right=(b[true_index+1]+truth)/2 if true_index<len(b)-1 else np.inf
                accuracy=float(ndtr((right-truth)/drift_sd)-ndtr((left-truth)/drift_sd))
                exact_accuracy=float(ndtr((right-truth)/exact_sd)-ndtr((left-truth)/exact_sd))
                theory_rows.append({"candidate_set":name,"truth":float(truth),"noise_sd":sigma,"feedback":g,"final_nearest_mean_distance":nearest_delta*shape[-1],"final_nearest_mahalanobis_sq":nearest_delta**2*shape[-1]**2/marginal_variance[-1],"cumulative_nearest_expected_score_margin":expected_margin,"analytic_accuracy":accuracy,"exact_analytic_accuracy":exact_accuracy,"analytic_drift_sd":float(drift_sd)})
                empirical=float(np.mean(selected==true_index))
                tolerance=6*np.sqrt(accuracy*(1-accuracy)/N)+2/N
                assert abs(empirical-accuracy) < tolerance
                tests.append({"candidate_set":name,"truth":float(truth),"sigma":sigma,"feedback":g,"analytic":accuracy,"empirical":empirical,"tolerance":float(tolerance),"pass":True})
write_csv("controlled_per_run.csv",all_rows)
write_csv("controlled_theory.csv",theory_rows)
write_csv("controlled_example_history.csv",history_rows)
pooled=[]
for name in C["candidate_sets"]:
    for sigma in C["observation_noise_sd"]:
        for g in C["feedback_strengths"]:
            group=[r for r in all_rows if r["candidate_set"]==name and r["noise_sd"]==sigma and r["feedback"]==g]
            row={"candidate_set":name,"noise_sd":sigma,"feedback":g,"n":len(group)}
            for key in arrays:
                vals=np.asarray([r[key] for r in group]);mean=float(vals.mean())
                row[key+"_mean"]=mean
                if key in ("correct","exact_correct"):
                    low,high=wilson(int(vals.sum()),len(vals));row[key+"_count"]=int(vals.sum())
                else:
                    half=1.959963984540054*float(vals.std(ddof=1))/np.sqrt(len(vals));low,high=mean-half,mean+half
                row[key+"_ci_low"],row[key+"_ci_high"]=float(low),float(high)
            pooled.append(row)
write_csv("controlled_summary.csv",pooled)
(OUT/"controlled_validation.json").write_text(json.dumps({"run_count":len(all_rows),"all_conditions_retained":True,"full_covariance_max_logscore_error":full_cov_max_error,"two_candidate_accuracy_checks":tests,"numpy":np.__version__},indent=2))
print(json.dumps({"rows":len(all_rows),"runtime_seconds":time.perf_counter()-start,"full_covariance_max_logscore_error":full_cov_max_error}),flush=True)
