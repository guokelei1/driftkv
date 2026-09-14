"""Executable candidate-response policy with the same bounded state machine.

Only the returned score is selected here. The base runner still performs the
real installation at the request boundary, before subsequent writes/requests.
"""
import argparse,json
from design2 import run_bounded as b

observe=b.observation;replay_user=b.replay.replay_user;latest=None
def capture(uid,target,stamp,panel,z,w,kind,state):
    global latest
    latest=(uid,target,stamp,w)
    observe(uid,target,stamp,panel,z,w,kind,state)

def replay_with_candidate(*args,**kwargs):
    original=b.replay.HOOK
    def select(*h):
        z=original(*h);key=(h[5],h[6])
        if key in b.PENDING:
            assert latest[:3]==(h[5],h[6],h[7])
            return latest[3]
        return z
    b.replay.HOOK=select
    try:return replay_user(*args,**kwargs)
    finally:b.replay.HOOK=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--uids-file');c=p.parse_args();c.variant='bounded'
    b.observation=capture;b.replay.replay_user=replay_with_candidate;b.main(c)
    path=b.r.ROOT/'results/design2'/c.run_id/'configuration.json';cfg=json.loads(path.read_text());cfg.update(response_rule='candidate_on_actual_commit',candidate_runner_sha256=b.r.base.sha(__file__));b.save(path,cfg)
