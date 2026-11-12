"""Compare reproduced numerical results with the supplied reference tables."""
from pathlib import Path
import argparse
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--controlled-root',type=Path,default=ROOT/'generated_controlled')
parser.add_argument('--figure-root',type=Path,default=ROOT/'generated_figures')
parser.add_argument('--statistics-root',type=Path,default=ROOT/'generated_statistics')
parser.add_argument('--output',type=Path,default=ROOT/'reproduction_check.json')
args=parser.parse_args()
checks=[]
for directory,names in [(args.controlled_root,['controlled_summary.csv','controlled_theory.csv']), (args.figure_root,['fhn_absolute_summary.csv','pk_45_summary.csv']), (args.statistics_root,['fhn_paired_primary.csv','fhn_paired_secondary.csv'])]:
    for name in names:
        expected=pd.read_csv(ROOT/'reference_results'/name)
        actual=pd.read_csv(directory/name)
        actual=actual[expected.columns]
        pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-10,atol=1e-12)
        checks.append({'file':name,'rows':len(actual),'matches_reference':True})
result={'all_passed':True,'numeric_relative_tolerance':1e-10,'numeric_absolute_tolerance':1e-12,'checks':checks}
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
