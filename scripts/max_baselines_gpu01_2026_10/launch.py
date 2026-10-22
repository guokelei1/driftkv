#!/usr/bin/env python3
"""Launch requested Max edges in a detached tmux; call only on user instruction."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from max_baselines_gpu01_2026_10.dual_gpu import b,binding_at,sources

BASE=ROOT/'experiments/01_quality_cost/runs/yambda/max'
PREPARED=BASE/'baselines_v0_v1_gpu01_202/attempt_1'


def run_manifest(path):
    manifest=json.loads(path.read_text())
    assert manifest['execution_source_sha256']==sources()
    lock=(BASE/'.gpu0_1.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',pid=os.getpid(),edges=manifest['edges'])
    child=None
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    try:
        for edge in manifest['edges']:
            out=Path(manifest['outputs'][str(edge)])
            binding_at(out)
            assert b.sha256(out/'configuration.json')==manifest['binding_sha256'][str(edge)]
            if (out/'summary.json').exists():
                summary=json.loads((out/'summary.json').read_text())
                if (summary.get('status')=='complete' and summary.get('quality_users')==196363
                        and summary.get('configuration_sha256')==b.sha256(out/'configuration.json')):
                    continue
            state.update(edge=edge)
            b.write_json(path.parent/'progress.json',state)
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('dual_gpu.py')),
                 '--phase','evaluate','--output',str(out),'--resume']
            with (path.parent/f'v{edge-1}_v{edge}.log').open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                state.update(child_pid=child.pid,command=cmd)
                b.write_json(path.parent/'progress.json',state)
                if child.wait():raise RuntimeError(f'edge {edge} failed; inspect its log')
        state.update(status='complete')
        b.write_json(path.parent/'progress.json',state)
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges',nargs='+',type=int,choices=[1])
    p.add_argument('--run-manifest',type=Path)
    p.add_argument('--resume-manifest',type=Path)
    args=p.parse_args()
    if args.run_manifest:return run_manifest(args.run_manifest)
    if args.resume_manifest:
        path=args.resume_manifest.resolve()
        manifest=json.loads(path.read_text())
        assert manifest['execution_source_sha256']==sources()
    else:
        assert args.edges and len(set(args.edges))==len(args.edges),'only edge 1 (V0-to-V1) is authorized in this launch'
        queue=BASE/'queue_v0_v1_gpu01_202'
        attempt=1
        while (queue/f'attempt_{attempt}').exists():attempt+=1
        out=queue/f'attempt_{attempt}'
        out.mkdir(parents=True)
        outputs={str(e):str(PREPARED/f'edge_v{e-1}_v{e}') for e in args.edges}
        for v in outputs.values():binding_at(Path(v))
        manifest=dict(edges=args.edges,outputs=outputs,execution_source_sha256=sources(),
            binding_sha256={e:b.sha256(Path(v)/'configuration.json') for e,v in outputs.items()},
            physical_gpus=[0,1],created_utc=datetime.now(timezone.utc).isoformat())
        path=out/'configuration.json'
        b.write_json(path,manifest)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    gpu_rows=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=uuid','--format=csv,noheader'],text=True)
    assert not any(uuid.strip() in occupied for uuid in gpu_rows.splitlines()),'GPU0/1 occupied; do not start a duplicate or displace a process'
    # The queue holds a host-local lock throughout all selected edges.
    check=(BASE/'.gpu0_1.lock').open('w')
    fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check.close()
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--run-manifest',str(path)]
    script=path.parent/'launch.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+
        '\nexport CUDA_VISIBLE_DEVICES=0,1\nexport PYTORCH_ALLOC_CONF=expandable_segments:True\nexec numactl --cpunodebind=0 --membind=0 '+shlex.join(cmd)+
        ' >> '+shlex.quote(str(path.parent/'queue.log'))+' 2>&1\n')
    session='max-baselines'
    subprocess.run(['tmux','-L','yambdamax202','new-session','-d','-s',session,'bash',str(script)],check=True)
    b.write_json(path.parent/'launch.json',dict(command=cmd,script=str(script),tmux_socket='yambdamax202',
        tmux_session=session,physical_gpus=[0,1],log=str(path.parent/'queue.log'),
        resume_command=[sys.executable,str(Path(__file__).resolve()),'--resume-manifest',str(path)],
        attach_command='tmux -L yambdamax202 attach -t max-baselines'))
    print(json.dumps(dict(status='launched',manifest=str(path),edges=manifest['edges'])),flush=True)


if __name__=='__main__':main()
