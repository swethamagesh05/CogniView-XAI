"""Audit or explicitly rerun CogniView analyses; never train or regenerate data."""
import argparse, csv, hashlib, json, subprocess, sys, time
from pathlib import Path
from datetime import datetime
ROOT = Path(__file__).resolve().parent
ET = 'src/eye_tracking/'
E = 'results/xai_multimodal/'
T = 'results/xai_trimodal/'
P = 'results/plr_features/'
R = 'results/gradcam_corrected/'
# id, script, arguments, prerequisites, expected outputs, working directory
STEPS = []
def add(i, script, inputs, outputs, args=(), cwd='.'):
    STEPS.append((i, script, args, inputs, outputs, cwd))
ret = 'models/retina/best_model.pth'
et = [ET+'data/ad_eyetracking/synthetic/metadata.csv', ET+'checkpoints/best_ad_tracker.pth']
add(1,'src/retinal/faithfulness_corrected.py',[ret,'data/train.csv','data/test.csv','data/test_images'],[R+'faithfulness_summary.json',R+'faithfulness_scores.csv'],['--num_samples','20','--steps','20','--random_runs','3'])
add(2,ET+'shap_ad_explainer.py',et,['results/eye_tracking_shap/shap_feature_importance.csv'],cwd=ET)
add(3,ET+'integrated_gradients.py',et,['results/xai_eyetracking/global_feature_ranking.csv'],['--num_subjects','30'],ET)
add(4,ET+'faithfulness_eyetracking.py',et,['results/xai_eyetracking/eyetracking_faithfulness_scores.csv'],['--num_subjects','20'],ET)
add(5,'src/plr/step7a_plr_quality.py',['results/plr_raw'],[P+'plr_quality_report.csv'])
add(6,'src/plr/step7b_plr_features.py',['results/plr_raw',P+'plr_quality_report.csv'],[P+'plr_session_features.csv',P+'plr_feature_summary.csv'])
add(7,'src/plr/step7c_plr_visualization.py',[P+'plr_session_features.csv'],[P+'plr_feature_distribution.png',P+'plr_response_distribution.png'])
add(8,'src/xai/fusion_xai.py',['data/fusion/val.csv','models/fusion/fusion_model.pth'],['results/xai_fusion/fusion_xai_per_window.csv'],cwd='src/xai')
for start, kind, folder, model in [(9,'extended',E,'fusion_extended_model.pth'),(12,'trimodal',T,'fusion_trimodal_model.pth')]:
    base='fusion_'+kind
    inputs=[f'data/{base}/{s}.csv' for s in ('train','val','test')]+[f'models/{base}/{model}',f'models/{base}/feature_scaler.pkl',folder+'training_summary.json']
    for offset, action, output in [(0,'evaluate','evaluation_metrics.json'),(1,'explain','modality_contribution_summary.json'),(2,'ablation','ablation_summary.json')]:
        add(start+offset,f'src/{base}/{action}_{kind}.py',inputs,[folder+output])
add(15,'src/fusion_trimodal/extract_retinal_features.py',[ret,'data/train.csv','data/train_images'],['results/retinal_features/retinal_feature_summary.json','results/retinal_features/retinal_embeddings.npy'])
def present(p):
    x=ROOT/p
    return x.is_dir() and any(x.iterdir()) or x.is_file() and x.stat().st_size>0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--run',action='store_true',help='Explicitly rerun selected analyses; may overwrite their outputs')
    parser.add_argument('--only',nargs='+',type=int)
    a=parser.parse_args()
    if a.run and not a.only: parser.error('--run requires --only IDs')
    if a.only and set(a.only)-{s[0] for s in STEPS}: parser.error('Unknown step ID')
    selected=[s for s in STEPS if not a.only or s[0] in a.only]
    if a.list:
        for i,s,*_ in STEPS: print(f'{i:2} {s}')
        return 0
    out=ROOT/'results/final_integration'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    rows=[]
    for i,script,args,inputs,outputs,cwd in selected:
        missing=[p for p in [script,*inputs] if not present(p)]
        status='NOT_READY' if missing else 'READY'
        elapsed=0
        if a.run and not missing:
            t=time.monotonic()
            with (out/f'step_{i:02}.log').open('w',encoding='utf-8') as log:
                rc=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT/cwd,stdout=log,stderr=subprocess.STDOUT).returncode
            elapsed=round(time.monotonic()-t,2)
            status='EXECUTED' if rc==0 else f'FAILED_{rc}'
        absent=[p for p in outputs if not present(p)]
        if absent and status in ('READY','EXECUTED'): status='OUTPUT_MISSING'
        rows.append(dict(step=i,script=script,status=status,missing_inputs=missing,missing_outputs=absent,seconds=elapsed))
        print(f'{i:2} {status:16} {script}',flush=True)
        if missing: print('   Missing:',', '.join(missing))
        if absent: print('   Missing outputs:',', '.join(absent))
        if a.run and status!='EXECUTED': break
    summaries={}
    for folder in (R,E,T,'results/retinal_features/'):
        for path in (ROOT/folder).glob('*.json'):
            try: summaries[path.relative_to(ROOT).as_posix()]=json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError,OSError) as exc: summaries[path.relative_to(ROOT).as_posix()]={'read_error':str(exc)}
    check=summaries.get(R+'faithfulness_summary.json',{})
    retinal_full_run=all(check.get(k)==v for k,v in {'evaluated_images':20,'perturbation_steps':20,'random_runs_per_image':3,'target_consistency':True,'overlapping_image_ids':0}.items())
    hash_matches=present(ret) and hashlib.sha256((ROOT/ret).read_bytes()).hexdigest()==check.get('checkpoint_sha256')
    report={'mode':'rerun' if a.run else 'audit_only','steps':rows,'retinal_full_run_metadata_matches':retinal_full_run,'retinal_checkpoint_hash_matches':hash_matches,'clinical_validation':False,'limitations':['Audit checks file presence, not scientific validity or output freshness.','Real retinal and PLR data are not paired clinical fusion inputs.','Fusion is synthetic; test sessions were previously evaluated.','Original retinal classification split is unverified.'],'summaries':summaries}
    (out/'integration_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'pipeline_status.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Retinal full-run metadata:',retinal_full_run,'Checkpoint hash match:',hash_matches)
    print('Report:',out)
    return int(any(r['status'] not in ('READY','EXECUTED') for r in rows) or not retinal_full_run or not hash_matches)
if __name__=='__main__':
    sys.exit(main())
