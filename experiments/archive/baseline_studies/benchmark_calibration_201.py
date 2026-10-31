from pathlib import Path
import json,time,sys,os
root=Path('/data/gkl/work/evokv')
import torch
from hstu_kvcache.evaluation.baselines import dataset_path, capture, fit_modules, fit_nonlinear, save_modules
from hstu_kvcache.evaluation.history import load_histories_semijoin
from hstu_kvcache.evaluation.yambda import load_model
from hstu_kvcache.common.artifacts import write_json, sha256
out=root/'experiments/01_quality_cost/runs/yambda/large/calibration_speed/seed_17/attempt_1'
out.mkdir(parents=True,exist_ok=False)
prior=root/'experiments/01_quality_cost/runs/yambda/large/baselines_v0_v1/seed_17/attempt_1/configuration.json'
binding=json.loads(prior.read_text());cfg=binding['settings'];cfg['physical_gpu']=0
train=binding['roles']['fit'][:64];val=binding['roles']['validation']
write_json(out/'configuration.json',dict(settings=cfg,fit=train,validation=val,source=str(prior),scope='one real 64-user history calibration for a runtime coefficient; no quality evaluation'))
torch.set_num_threads(4);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
os.environ['EVOKV_ATTENTION_BACKEND']='triton'
device=torch.device('cuda:0');torch.cuda.set_device(device)
began=time.perf_counter();path=dataset_path('large');ds=json.loads(path.read_text())
histories=load_histories_semijoin(train+val,dataset_path=path,known_vocab_size=ds['foundation_items'],oov_buckets=256,start_timestamp=231*86400,end_timestamp=231*86400,max_history=1024,threads=8)
input_seconds=time.perf_counter()-began
base=root/'results/unified_training_2026_09/large/seed17/checkpoints'
began=time.perf_counter();old,raw=load_model(base/'v0/checkpoint_100.pt',device);del raw
current,raw=load_model(base/'v1/checkpoint_100.pt',device);del raw
old.eval().requires_grad_(False);current.eval().requires_grad_(False)
torch.cuda.synchronize();model_seconds=time.perf_counter()-began
began=time.perf_counter();rows,_,transfer=capture(old,current,histories,train+val,cfg,device)
torch.cuda.synchronize();capture_seconds=time.perf_counter()-began
began=time.perf_counter();modules,records,ledger=fit_modules(current,rows,train,val,cfg,device,'large')
modules,nonlinear=fit_nonlinear(current,rows,train,val,modules,config=cfg,device=device,scale='large',batch_size=8)
torch.cuda.synchronize();fit_seconds=time.perf_counter()-began
save_modules(out,'history_conditioned_fit64',modules,dict(method='history_conditioned',fit_users=train,validation_users=val,calibration_flops=ledger['calibration_flops']+nonlinear['cost']['calibration_flops'],details=dict(affine=records,affine_cost=ledger,nonlinear=nonlinear)))
write_json(out/'summary.json',dict(status='complete',input_seconds=input_seconds,model_load_seconds=model_seconds,capture_seconds=capture_seconds,fit_seconds=fit_seconds,users=64,validation_users=len(val),epochs=cfg['nonlinear_epochs'],physical_gpu=0,scope='one calibration timing coefficient, progress wall time not a paper timing result',weights_sha256=sha256(out/'history_conditioned_fit64/calibration.pt')))
print(json.dumps(dict(status='complete',capture_seconds=capture_seconds,fit_seconds=fit_seconds)),flush=True)
