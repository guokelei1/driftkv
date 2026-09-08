#!/usr/bin/env python3
"""CPU-only arithmetic ledger from frozen shapes, metadata and existing feedback.

No backbone inference on population, no fitting, no timing-to-FLOPs conversion.
Counts are scalar arithmetic under explicitly stated algorithms, not hardware instructions.
"""

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from design.data import DAY, ROOT, checkpoint, histories
from design.report_native_base import bootstrap, bootstrap_canary
from insight_two.common import CUTOVER_DAYS

D, L, H, d, R, V = 192, 6, 6, 32, 32, 2304
TARGETS = (1, 3, 4, 5)
METHODS = ("native", "summary_input", "no_source")
RUN = ROOT / "results/design/native_base_quality4091_01"
OUT = ROOT / "results/design/analysis/native_flops_01"
DESIGN = ROOT / "results/design"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                     default=lambda x:x.item())+"\n")


def rebuild(n):
    """Cache-only dependency closure: five complete blocks, final norm+K/V."""
    pairs = n*(n+1)//2
    gemm = 2*n*(32*D+D*D+27*D*D)+4*5*D*pairs
    # RMS: square, sum, /D, +eps, two vector products =4D+1; rsqrt separate.
    # Input: phase multiplication and two additions. Block: gate product/residual.
    scalar = n*(16+2*D+L*(4*D+1)+5*2*D)+2*5*H*pairs
    return dict(gemm=gemm, scalar=scalar, elu=5*H*pairs, silu=5*n*D,
                rsqrt=L*n, sincos=32*n)


def value(ops, special=1):
    return ops["gemm"]+ops["scalar"]+special*sum(ops[k] for k in ("elu", "silu", "rsqrt", "sincos"))


def prefix_literal(n):
    """Observed compute_kv eager graph, reported separately; never main denominator."""
    pairs=n*n
    return dict(gemm=2*n*(32*D+D*D+30*D*D)+4*L*D*pairs,
                scalar=n*(16+2*D+7*(4*D+1)+6*4*D)+3*L*H*pairs,
                elu=L*H*pairs,silu=L*n*D,rsqrt=7*n,sincos=32*n)


def append_ops(n, m):
    # Actual rectangular rolling-band kernel; n retained before append.
    # Final hidden is returned by append interface; all six blocks are computed.
    pairs=m*(n+m)
    return dict(gemm=2*m*(32*D+D*D+30*D*D)+4*L*D*pairs,
                scalar=m*(16+2*D+6*(4*D+1)+6*4*D)+3*L*H*pairs,
                elu=L*H*pairs,silu=L*m*D,rsqrt=6*m,sincos=32*m)


def query_ops(n, q):
    return dict(gemm=q*(2*(32*D+D*D+30*D*D)+4*L*D*n+2*D),
                scalar=q*(16+3*D+7*(4*D+1)+L*(6*D+H+2*H*n)+1),
                elu=L*H*q*(n+1),silu=L*D*q,rsqrt=7*q,sincos=32*q)


def read_cost(q, calls):
    # Per candidate: U GEMM, head affine GEMM, normalization and vector additions.
    # Per request group: scale stored b/A by N (as literal reader does).
    return q*(2*L*D*D+2*L*H*d*d+7*L*D)+calls*L*(D+H*d*d+1)


def prepare_cost(t, segments, no_source=False):
    p=t+1
    s=p*(V+1)+1
    # SummaryWriter.pack: payload divide+mask; ordinal/timestamp divisions.
    pack=segments*(2*V+2)
    # features: mass, weighted sum /total, fraction for every producer, age.
    features=p*(2*segments*V+2*segments+1)+segments+1
    pca=2*s+2*s*R
    # Active mask comparisons/integer bookkeeping separately, one float age field.
    view=2*L*H*(1 if no_source else R+1)*(d+1)*d
    view+=L*(H*d*d+2*H*d*d+H*d+H*d+H*d*d)
    means=p*(2*segments*V+2*segments+1)
    return dict(pack=pack,features=features,pca=pca,view=view,means=means)


def solver(s,q,rank):
    """fit_joint block contractions; LU solve, not dense design Gram/Cholesky.

    Three-input einsums use their lower-intermediate contraction (opt_einsum).
    Include residual validation and training-prediction diagnostics actually executed.
    Elementwise preprocessing is itemized as a separate explicit budget.
    """
    j=d+1
    p=rank*j+D
    parts={
        "qgram":2*s*H*j*q*j,
        "source_outer":s*rank*rank,
        "gram_contract":2*rank*rank*s*H*j*j,
        "qty_rhs":2*s*H*j*q*d+2*H*rank*s*j*d,
        "qta_cross":2*s*H*j*q*D+2*H*rank*s*j*D,
        "read_gram":2*s*q*D*D,
        "read_rhs":2*H*D*s*q*d,
        "LU_and_solve":H*((2/3)*p**3+2*p*p*d),
        "residual_check":2*H*p*p*d,
        "prediction":2*s*rank*H*j*d+2*s*H*q*j*d+2*s*q*D*D,
        "scalar_preprocessing":(8*s*H*q*d+5*s*q*D+3*s+H*p*p+
                                  s*H*j*j+s*H*j*d+s*H*j*D+D*D+H*D*d),
    }
    return {k:L*v for k,v in parts.items()}


def shapes_and_canary():
    shapes={}
    for t in TARGETS:
        p=torch.load(DESIGN/f"native_coverage384_01/translator_C_m{t}.pt",map_location="cpu",weights_only=True)
        src=torch.load(DESIGN/f"mechanism_aggregate_factorial192_01/source_projection_mean_m{t}.pt",map_location="cpu",weights_only=True)
        assert len(p)==L and src["projection"].shape==((t+1)*(V+1)+1,R)
        assert all(x["weights"].shape==(33,H,33,d) and x["read_weights"].shape==(H,D,d) for x in p)
        shapes[str(t)]=dict(source_dim=src["projection"].shape[0],weights=list(p[0]["weights"].shape),read_weights=list(p[0]["read_weights"].shape))
    # Only checkpoint metadata/matrix shapes; no population model execution.
    payload=torch.load(checkpoint(1),map_location="cpu",weights_only=False,mmap=True)
    config=payload["config"]
    from hstu_kvcache.models.hstu import HSTU, HSTUConfig
    cfg=dict(config)
    cfg.update(num_items=8,num_prediction_items=8)
    cfg={k:v for k,v in cfg.items() if k in HSTUConfig.__dataclass_fields__}
    assert (cfg["hidden_size"],cfg["num_layers"],cfg["num_heads"])==(D,L,H)
    assert cfg["block_variant"]=="legacy" and cfg["activation"]=="elu_plus1"
    torch.manual_seed(17)
    model=HSTU(HSTUConfig(**cfg)).double().eval()
    items=torch.tensor([[1,2,3,4]])
    actions=torch.ones_like(items)
    delta=torch.tensor([[0.,1.,2.,3.]],dtype=torch.double)
    with torch.no_grad():
        expected=model.compute_kv(items,actions,delta)
        x=model.embed_inputs(items,actions,delta)
        kv=[]
        for layer,block in enumerate(model.blocks):
            norm=block.norm(x)
            k,v=block.attn.k_proj(norm),block.attn.v_proj(norm)
            kv.append((k,v))
            if layer<L-1:
                x=block(x)
        torch.testing.assert_close(torch.stack([v[0] for v in kv]),expected.k,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(torch.stack([v[1] for v in kv]),expected.v,rtol=1e-12,atol=1e-12)
        # Fold native normalization /count and mean into T and intercept.
        p=torch.load(DESIGN/"native_coverage384_01/translator_C_m1.pt",map_location="cpu",weights_only=True)[0]
        native=torch.randn(7,D,dtype=torch.double); n=1024.
        old=torch.einsum("qa,had->hqd",(native/n-p["read_center"])/p["read_scale"],p["read_weights"])*n
        t=p["read_weights"]/p["read_scale"][None,:,None]
        bias=torch.einsum("a,had->hd",p["read_center"],t)
        new=torch.einsum("qa,had->hqd",native,t)-n*bias[:,None]
        torch.testing.assert_close(old,new,rtol=1e-11,atol=1e-11)
    bootstrap_canary()
    dump("shapes_canary.json",dict(shapes=shapes,config=config,cache_closure="4-token synthetic equality passed",native_fold="FP64 identity passed",auc="weighted tie reference passed"))


def population_counts(raw):
    states=json.loads((RUN/"state_counts.json").read_text())
    result=[]
    for t in TARGETS:
        f=raw[raw.target==t]
        groups=f.groupby(["uid","timestamp"]).agg(writes=("writes_since_release","first"),q=("uid","size")).reset_index()
        assert (f["count"]==1024).all()
        groups=groups.sort_values(["uid","timestamp"])
        previous=groups.groupby("uid").writes.shift().fillna(0)
        dirty=groups.writes>previous
        # Initial parent occupies exactly one closed segment. New producer segments
        # are 1024 events each; rolling1024 retains 1 segment on boundaries, else2.
        refresh_segments=np.where(groups.loc[dirty,"writes"].to_numpy()%1024==0,1,2)
        r=[x for x in states if x["target"]==t]
        assert len(r)==4091 and sum(x["requests"] for x in r)==len(f)
        assert sum(x["checks"]["unchanged_scoring_states"] for x in r)==len(groups)
        assert all(x["events"]==x["counts"]["appends"]==x["counts"]["evictions"] for x in r)
        result.append(dict(target=t,users=4091,requests=len(f),read_calls=len(groups),
            appends=sum(x["events"] for x in r),evictions=sum(x["counts"]["evictions"] for x in r),
            publications=4091,refreshes=int(dirty.sum()),refresh_segments=int(refresh_segments.sum()),
            no_feedback=sum(x["requests"]==0 for x in r),
            active_layer_queries=sum(int(((f.writes_since_release.to_numpy()<1024*(l+1))).sum()) for l in range(L))))
    dump("population_counts.json",result)
    return result


def calibration_counts(history, fit, life):
    """Replay ONLY integer lengths/timestamps from existing calibration protocol."""
    builds,appends=defaultdict(Counter),defaultdict(Counter)
    scene_counts={}
    allocation={str(t):dict(builds=defaultdict(Counter),appends=defaultdict(Counter)) for t in TARGETS}
    charge=1
    def end(uid,ts): return int(np.searchsorted(history.rows[uid][0],ts,side="left"))
    def build(uid,ts,kind):
        n=min(1024,end(uid,ts)); builds[kind][n]+=1
        allocation[str(charge)]["builds"][kind][str(n)]+=1
        return n
    def replay(n,e,kind):
        for offset in range(0,e,128):
            m=min(128,e-offset); appends[kind][n,m]+=1
            allocation[str(charge)]["appends"][kind][f"{n},{m}"]+=1
            n=min(1024,n+m)
        return n
    start=CUTOVER_DAYS[0]*DAY
    early, lengths={},{}
    for u in fit:
        n=build(u,start,"initial_source")
        ts=history.rows[u][0]; k=end(u,start)
        cut=int(ts[k-4]) if n>4 and ts[k-4]>ts[k-n] else start
        old=build(u,cut,"initial_source")
        early[u]=(old,k-end(u,cut),cut)
        lengths[u]=n
    for t in range(1,6):
        charge=t
        cut=CUTOVER_DAYS[t-1]*DAY
        if t!=2:
            sizes=[]
            for u in fit:
                old,e,snapshot=early[u]
                sizes.extend([lengths[u],min(1024,old+e)])
                points=[]
                if u in life:
                    ts=history.rows[u][0]; k=end(u,cut)
                    points=list(dict.fromkeys(int(ts[i]) for tail in (64,256,1024,2048,4096,6144) if (i:=max(1024,k-tail))<k))
                for p in dict.fromkeys([cut,snapshot,*points]):
                    build(u,p,"teacher")
                replay(min(1024,end(u,snapshot)),e,"teacher_early")
                replay(old,e,"source_early_training")
                if t>1:
                    build(u,cut,"adjacent_source")
                    sizes.append(min(1024,end(u,cut)))
                for p in points:
                    n=build(u,p,"lifetime_source"); events=end(u,cut)-end(u,p)
                    replay(n,events,"lifetime_source")
                    replay(n,events,"lifetime_teacher")
                    sizes.append(min(1024,n+events))
                if u in life:
                    sizes.append(min(1024,end(u,cut)))
            scene_counts[str(t)]={"scenes":len(sizes),"sum_lengths":sum(sizes)}
            meta=pd.read_parquet(DESIGN/f"native_coverage384_01/scenes_m{t}.parquet")
            actual=meta[meta.uid.isin(fit)]
            assert len(sizes)==len(actual) and sum(sizes)==int(actual["count"].sum()), (t,len(sizes),len(actual))
        if t<5:
            # Charge preparatory lineage to the next actual adaptation target;
            # M1→M2 and M2→M3 both prepare M3; no synthetic M2 adaptation row.
            charge={1:3,2:3,3:4,4:5}[t]
            stop=CUTOVER_DAYS[t]*DAY
            for u in fit:
                ts=history.rows[u][0]; left,right=end(u,cut),end(u,stop)
                snap=int(ts[max(left,right-4)]) if right>left else stop
                middle=end(u,snap)
                n=replay(lengths[u],middle-left,"lineage")
                early[u]=(n,right-middle,snap)
                lengths[u]=replay(n,right-middle,"lineage")
    return dict(by_target=allocation,scene_validation=scene_counts,builds={k:{str(n):c for n,c in v.items()} for k,v in builds.items()},
                appends={k:{f"{n},{m}":c for (n,m),c in v.items()} for k,v in appends.items()})


def shared_cost(counts, old_counts, method, special=1, eigen_coefficient=9):
    details={}
    for label,c in (("C256_preparation",counts),("A64_coordinate_dependency",old_counts)):
        # Existing preparation genuinely calls full eager compute_kv and rectangular
        # rolling replay. Also provide a dependency-pruned counterfactual separately.
        details[label+"_prefix"]=sum(value(prefix_literal(int(n)),special)*v for hist in c["builds"].values() for n,v in hist.items())
        details[label+"_replay"]=sum(value(append_ops(*map(int,n.split(','))),special)*v for hist in c["appends"].values() for n,v in hist.items())
        tokens=sum(int(n)*v for k,hist in c["builds"].items() if k!="teacher" for n,v in hist.items())
        tokens+=2*sum(int(n.split(',')[1])*v for k,hist in c["appends"].items() if "teacher" not in k for n,v in hist.items())
        details[label+"_summary_maintenance"]=V*tokens
    solvers={}
    for t in TARGETS:
        meta=pd.read_parquet(DESIGN/f"native_coverage384_01/scenes_m{t}.parquet")
        fits=meta[meta.group.isin(["original_fitting","additional_fitting"])]
        old=meta[meta.group=="original_fitting"]
        for label,f,q,rank,kind in (("C",fits,16,1 if method=="no_source" else 33,method),
                                   ("A_coordinates",old,64,33,"native")):
            n=f["count"].to_numpy(dtype=int); s=len(n)
            # Actual six full query passes (one acquisition pass per fitted layer),
            # plus 6 target-history responses; 15 earlier fitted correction layers.
            forward=sum(6*value(query_ops(int(x),q),special) for x in n)
            teacher=sum(L*q*(4*D*int(x)+2*H*int(x)+special*H*int(x)+2*D) for x in n)
            partial=15*s*q*(2*D*D+2*H*d*d+7*D)
            extra=6*L*s*q*((4*D+(3+special)*H)*(t+1)) if kind=="summary_input" else 0
            # Per-call scaling existing b/A even on layers awaiting a fitted U.
            forward+=36*s*(D+H*d*d+1)
            details[f"M{t}_{label}_response_collection"]=forward+teacher+partial+extra
            sol=solver(s,q,rank)
            solvers[f"M{t}_{label}"]=sol
            details[f"M{t}_{label}_solver"]=sum(sol.values())
            # Scene pack/source encode and installed views; <=8 real producer segments
            # from six producers. Use retained-state metadata bound; explicitly bounded.
            prep=prepare_cost(t,8,rank==1)
            details[f"M{t}_{label}_state_encode_view_bound"]=s*sum(prep[k] for k in ("pack","features","pca","view"))
        s=len(old); dim=(t+1)*(V+1)+1
        # source_projection uses Gram eigendecomposition, not randomized PCA.
        # 9s^3 is declared symmetric eigensolver leading-order allowance, varied later.
        details[f"M{t}_PCA_coordinate_dependency"]=2*s*s*dim+eigen_coefficient*s**3+2*dim*s*R+8*s*dim+2*dim*R
    # Frequency construction on every temporal call, pessimistically unbatched;
    # centers/scales in A: mean/std including roots, masks/reductions and pack metadata.
    calls=sum(v for c in (counts,old_counts) for kind in ("builds","appends") for hist in c[kind].values() for v in hist.values())
    calls+=sum(pd.read_parquet(DESIGN/f"native_coverage384_01/scenes_m{t}.parquet").group.isin(["original_fitting","additional_fitting"]).sum()*36 for t in TARGETS)
    details["frequency_and_coordinate_scalar_allowance"]=int(calls)*(64+16*special)
    return details,solvers


def population_cost(rows,method):
    totals=Counter()
    for r in rows:
        t=r["target"]; p=t+1
        totals["init"]+=r["users"]*V*(1024-1)
        totals["maintain"]+=V*(r["appends"]+r["evictions"])
        for label,cnt,segments in (("publish",r["publications"],r["publications"]),
                                   ("refresh",r["refreshes"],r["refresh_segments"])):
            # Linear in number of retained segments, split from fixed per-call work.
            one=prepare_cost(t,1,method=="no_source"); two=prepare_cost(t,2,method=="no_source")
            keys=("pack","features","pca","view")+(("means",) if method=="summary_input" else ())
            for key in keys:
                totals[label+"_"+key]+=cnt*one[key]+(segments-cnt)*(two[key]-one[key])
        totals["read"]+=read_cost(r["requests"],r["read_calls"])
        if method=="summary_input":
            totals["read"]+=r["requests"]*L*((4*D+3*H)*p+H*p)  # special ELU=1
    return dict(totals)


def render(result,rows,raw,boot):
    from hstu_kvcache.evaluation.binary_metrics import binary_metrics
    quality={int(t):{m:binary_metrics(f.label.to_numpy(),f[m].to_numpy())["ROC_AUC"]
                     for m in ("reuse","exact",*METHODS)} for t,f in raw.groupby("target")}
    def interval(target,left,right):
        v=next(v for v in boot["differences"] if v["target"]==target and v["left"]==left and v["right"]==right)
        return "["+", ".join(f"{100*x:+.4f}" for x in v["interval"])+"]"
    base=rebuild(1024)
    lines=["# 读取修正的增量FLOPs与人口规模分析", "",
        "冻结C、summary_input、no_source；只读4091人既有反馈/状态计数、256校准历史时间戳与冻结参数。无新拟合、无新人口前向、无kernel优化；确认6000未读。", "",
        "## 1. 核算口径与结论", "",
        "Reuse增量=0；共同Parent初始化、正常读取与同形状native追加抵消。主分母使用裁去无用末层输出、只算有效因果配对的Current KV依赖闭包。方法端保留当前校准的较贵完整prefix与矩形回放，包含生成冻结PCA和A坐标所需的64用户依赖；这是一份保守的明确算法预算，不是从墙钟拟合的模型。", "",
        "GEMM=2mkn，基本加减乘除各1。ELU/SiLU/rsqrt/sin/cos调用单列，主表每次折1个约定操作；这不是硬件指令数，后表将折算提高到8/32检验稳健性。eigh实现依赖迭代与库，采用9s³的显式主阶预算并以20s³复核；不声称是逐指令精确计数。校准源摘要使用最多8段的上界；基本归一化/统计按显式标量预算收费。全部公式和计数保留于生成脚本/JSON，表中小数不代表测量精度。", "",
        "**在固定256校准、当前成熟域与评分/事件强度假设下，完整native在3万用户已具有明显增量FLOPs优势。优先删除用户状态路径没有得到本账本支持。** 小人口共享准备昂贵；扩大人口只摊薄K，不能消除每用户/每请求费用。墙钟821.29s/577.53s继续是实现尚未节省时间的证据，不否定理论算术优势。", "",
        "## 2. Exact依赖与服务算子表", "",
        "真实接口为data.prefix取最后min(历史,1024)条，首个时间差归零，再compute_kv；没有隐藏更长prefix重放。此为现有单边初始语义，不等于持续历史的全局Exact。每层K/V都在attention之前，故末层只需RMSNorm与K/V，前五层必须完整保留。", "",
        "| 算子 | 矩阵/公式 | 次数/共享范围 | Reuse公共项 |", "| --- | --- | --- | --- |",
        "| Exact输入 | 时间32×192、输入192×192；2N(32D+D²) | 每用户/更新 | 否 |",
        "| Exact前五层 | Q/K/V/out/gate均192×192；2N·25D² | 每用户/更新 | 否 |",
        "| Exact第六层 | K/V两个192×192；2N·2D² | 每用户/更新 | 否 |",
        "| Exact聚合 | 4·5D·N(N+1)/2；ELU×5h·N(N+1)/2 | 有效下三角 | 否 |",
        "| Exact逐元素 | 6N(4D+1) RMS基本操作、5N gate/residual、输入与attention缩放 | 特殊调用另列 | 否 |",
        "| 普通native读取/追加 | 同形状、同事件次数 | 全服务期 | 是；数值不同不影响FLOPs抵消 |",
        "| U响应变换 | 2LD²=442368 | 每候选；三方法相同 | 否；a_source本身已在公共读取里 |",
        "| q仿射 | 2Lh d²=73728，加截距；增广写法76032 | 每候选 | 否 |",
        "| 原读入归一化/缩放 | 7LD/候选；L(D+hd²+1)/请求组 | 即使mask=0，原代码仍执行 | 否 |",
        "| 源条件视图 | 2Lh(r+1)(d+1)d=2509056；再还原query坐标、mask | 每发布/dirty刷新 | 否 |",
        "| 无源条件视图 | 同式r+1=1；原代码仍重复安装 | 每发布/dirty刷新 | 否 |",
        "| 源投影 | (s−center)/scale、2s·32；s=4611/9221/11526/13831 | 每发布/dirty刷新 | 否 |",
        "| 摘要首次构建 | 2LD(N−1) | 四个独立初始状态各一次 | 否 |",
        "| 摘要维护 | 2304/追加，2304/淘汰 | 全部真实事件，无反馈也维护 | 否 |",
        "| 摘要输入额外响应 | (4D+3h)P/层/候选，加hP个ELU调用；P=target+1 | 包括零质量producer槽，原代码不跳过 | 否 |", "",
        f"N1024每用户每次Exact：GEMM {base['gemm']/1e9:.9f} GF，基本逐元素{base['scalar']/1e9:.9f} GF；ELU {base['elu']:,}、SiLU {base['silu']:,}、rsqrt {base['rsqrt']:,}、sin/cos {base['sincos']:,}次。主折算每次{value(base)/1e9:.9f} GF，四次{4*value(base)/1e9:.9f} GF。",
        "原eager compute_kv还执行第六层完整attention、final_norm及密集上三角；其算子计数仅用于当前校准路径，不用来放大人口Exact分母。4-token合成canary确认裁剪后全部六层K/V与原compute_kv相同。", "",
        "## 3. 真实生命周期次数", "",
        "| M | 用户/初始化/发布 | 候选评分 | 请求时间组 | dirty刷新 | 写入=淘汰 | 无反馈UID |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| M{r['target']} | {r['users']} | {r['requests']} | {r['read_calls']} | {r['refreshes']} | {r['appends']} | {r['no_feedback']} |")
    lines += ["", "原state_counts.refreshes属于旧入口，值0不能用于新NativeRelease.prepare。按每UID时间组的writes_since_release严格增加恢复dirty次数；首次写后评分即刷新，末尾无评分不刷新。新旧计数与全部请求、普通追加/淘汰、unchanged_scoring_states逐项一致。初始1段，1024窗口在写入边界1段、非边界2段，刷新段数由写入序号恢复。", "",
        "四次独立初始化各收backfill，不将其当成连续发布重复构建，也不假设先前已经免费维护。88860条标签反馈仅提供本trace候选调用数量，不代表生产全候选。", "",
        "## 4. 共享校准K：计入冻结依赖", "",
        "当前C需要256用户源/教师、原生谱系与生命周期辅助状态；仍经过M2普通过渡但不拟合M2。A64是冻结query/native坐标的生成依赖，需A的逐层拟合及源PCA；这些不是免费文件。与C准备重复的A成本在本保守账本中仍收取，不把历史研究所有失败实验/诊断用户教师都转嫁给部署。", "",
        "| 方法 | 源/教师prefix TF | 校准回放 TF | 响应收集 TF | 联合求解 TF | 其他/坐标 TF | K总TF |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m,v in result.items():
        parts=v["shared_FLOPs"]
        sums=[sum(x for key,x in parts.items() if key.endswith(suffix)) for suffix in ("_prefix","_replay","_response_collection","_solver")]
        lines.append(f"| {m} | "+" | ".join(f"{x/1e12:.4f}" for x in [*sums,v['K']-sum(sums),v['K']])+" |")
    lines += ["", "当前fit_joint先构造source_outer、query Gram，再做跨场景收缩；read Gram在heads间共享，cross/rhs单列。联合特征p=1281或225，torch.linalg.solve按LU主阶(2/3)p³+2p²d/头核算，另外收残差核对、训练预测与标量预处理。绝不使用2mp²的完整展开Gram冒充实际分块算法，也不称其Cholesky。完整逐收缩公式见solver_FLOPs字段及solver()。", "",
        "PCA以旧拟合场景x xᵀ求eigh，再xᵀV构造投影，非随机SVD；中心/尺度和投影计算计费。拟合场景M1/M3/M4/M5为747/1023/1029/1035；A依赖191/257/258/258。时间戳重建的场景数量与长度总和均与留存metadata一致，所有UID保留。", "",
        "## 5. 人口外推与目标阈值", "",
        "Exact(U)=Ue，Method(U)=K+Ua；rho_life=a/e+K/(Ue)。发布只收K+init+publish。下表沿用4091人的历史长度、事件/候选/dirty分布，固定256校准；百万规模是计算场景假设，不是质量已验证。", "",
        "| 方法 | 人口 | Exact增量TF | 方法增量TF | 发布比例 | 生命周期比例 | 生命周期节省 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m,v in result.items():
        for r in v["scales"]:
            lines.append(f"| {m} | {r['users']:,} | {r['exact_TF']:.3f} | {r['method_TF']:.3f} | {r['release_ratio']:.2%} | {r['life_ratio']:.2%} | {r['saving']:.2%} |")
    lines += ["", "| 方法 | a GF/用户/四窗口 | 渐近生命周期比例a/e | 生命周期持平人口 | 发布20%人口 | 生命周期20%人口（仅辅助） |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m,v in result.items():
        th=v["thresholds"]
        lines.append(f"| {m} | {v['a']/1e9:.6f} | {v['asymptotic_life_ratio']:.3%} | {th['life_break_even']:.0f} | {th['release_20pct']:.0f} | {th['life_20pct_optional']:.0f} |")
    lines += ["", "U_beta=K/(beta e−a)，分母≤0则无法靠扩大人口达成。发布用对应a_release替换a。这里生命周期20%仅补充计算，不设为新研究门槛。K主要是校准状态构建/回放，不是用户状态视图，也不是回归参数数量；不据此启动减校准新实验。", "",
        "## 6. 质量与理论计算同表", "",
        "质量全部沿用原4091成熟用户；下面3万人成本不意味着3万人重新测过质量。新增Exact配对差来自相同4091UID、四边同权重的1000次seed17 bootstrap。", "",
        "| 方法 | 原质量相对Reuse pp | 原质量相对Exact pp [描述区间] | 3万生命周期比例 | 10万比例 | 100万比例 |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for m,v in result.items():
        dr=np.mean([q[m]-q['reuse'] for q in quality.values()])*100
        de=np.mean([q[m]-q['exact'] for q in quality.values()])*100
        lines.append(f"| {m} | {dr:+.4f} | {de:+.4f} {interval('equal_edge',m,'exact')} | "+" | ".join(f"{r['life_ratio']:.2%}" for r in v['scales'][1:])+" |")
    lines += ["", "| M | native−Exact pp [区间] | 摘要−Exact pp [区间] | 无源−Exact pp [区间] |", "| --- | --- | --- | --- |"]
    for t in TARGETS:
        lines.append(f"| M{t} | "+" | ".join(f"{100*(quality[t][m]-quality[t]['exact']):+.4f} {interval(t,m,'exact')}" for m in METHODS)+" |")
    lines += ["", "native/摘要的汇总Exact差区间含0不等于质量等价；无源状态汇总略低于Exact的描述区间也不证明源条件必需（直接native−无源区间仍含0）。Exact不是反馈AUC的数学上界；native的教师距离优势不能改写成已确认的AUC必要性。", "",
        "## 7. 评分量、长度与计数约定敏感性", "",
        "下表完整native、3万人；固定真实事件率。左列同一缓存状态增加评分，不额外刷新；右列每个新增调用至多增加一次dirty刷新，且以每次真实写入最多一次刷新封顶，是保守调度情景，不是重放过的生产trace。", "",
        "| 评分倍数 | 每用户四窗口候选 | 同刷新生命周期比例 | 额外dirty上界比例 |", "| --- | ---: | ---: | ---: |"]
    for r in result['native']['query_sensitivity']:
        lines.append(f"| {r['query_multiplier']} | {r['queries_per_user']:.2f} | {r['life_ratio_30k_same_refresh']:.2%} | {r['life_ratio_30k_extra_dirty_bound']:.2%} |")
    lines += ["", "| 假设历史长度 | native 3万生命周期比例 |", "| --- | ---: |"]
    for r in result['native']['length_sensitivity']:
        lines.append(f"| {r['history_length']} | {r['life_ratio_30k']:.2%} |")
    lines += ["", "长度表保持当前K、事件/评分/刷新次数，只改变Exact依赖量和backfill，是条件成本曲线；并非新长度上的质量评测，不能据此设置短历史Exact阈值。若重新设计窗口/校准，必须重算K和实际轨迹。", "",
        "| 特殊函数折算/次 | eigh立方项系数 | native 3万生命周期比例 |", "| --- | ---: | ---: |"]
    for r in result['native']['numerical_convention_sensitivity']:
        lines.append(f"| {r['special_function_weight']} | {r['eigen_cubic_allowance']} | {r['life_ratio_30k']:.2%} |")
    lines += ["", "ELU调用不是必定执行exp的次数；输入正负分布未重读，因此不伪报精确exp数，按最多每ELU一次非线性调用预算。频率生成与A坐标统计另有保守标量余量。即使特殊函数和eigh系数提高，本人口结论不由这些低阶计数决定。", "",
        "## 8. 有等价依据的候选账本与存储边界", "",
        "仅做公式和合成数值核对，不修改服务代码：令T=diag(1/read_scale)U，则N*((a_source/N−center)/scale)U = a_source T−N·center T。后项吸收进状态截距；query中心/尺度可在发布时折入W。mask仍作用于历史修正，self、projection、gate与残差顺序不变。FP64代数canary通过，不宣称FP32舍入逐位相同。", "",
        "无源状态b/A本来只依版本，可发布一次生成，保留N和年龄mask；此候选能删除其不需要的源摘要/PCA/dirty视图计算。完整/摘要仍需自己的状态视图，不把其成本一起删掉。三者仍保留原K预算，未假设新拟合或更便宜教师。", "",
        "| 方法 | 当前用户侧GF/用户 | 等价候选GF/用户 | 候选3万生命周期比例 |", "| --- | ---: | ---: | ---: |"]
    for m,v in result.items():
        c=v['candidate_equivalent']
        lines.append(f"| {m} | {v['a']/1e9:.6f} | {c['a']/1e9:.6f} | {c['life_ratio_30k']:.2%} |")
    lines += ["", "这份候选只展示可解释的算术差，不部署、不测新质量、不用候选账本替代当前账本。3万人已明显有优势，无需为此急删状态路径；未来若减少K，应先查源/教师prefix与生命周期回放，不能仅减W/U。", "",
        "普通KV FP32为6×2×N×192×4字节，N1024为9 MiB/用户；额外summary为每段9216字节，当前1–2段；就绪b/A为6×6×33×32×4=152064字节/用户/方法。源PCA FP64每发布s×32×8字节，W/U分别1475712/259200个系数，独立列为共享存储而非读时FLOPs。服务原型每次评分仍复制/索引/安装张量，0 FLOPs不意味0时间。", "",
        "Backfill至少读一次普通KV；每次写入/淘汰摘要各读取一个事件的2304浮点值，每次dirty要读段摘要并输出152064字节视图。实际访存含中间张量与重复小算子，未以这些最小字节量冒充实测带宽；metadata整数操作和调用量见population_counts。I/O、同步、调度、内存容量是下一实现阶段问题。", "",
        "## 9. 验证、证据与研究决定", "",
        "生成：`PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/design/report_native_flops.py`。校准只扫描已有合法256UID时间戳；场景数/长度与冻结metadata交叉验证，模型canary只用4个合成token；原质量不变，额外bootstrap不调用模型。原始运行哈希及本次输入/代码哈希见provenance.json。", "",
        "交付：ledger.json完整三方法/候选/阈值/敏感性；calibration_counts.json构建长度与回放形状直方图；population_counts.json真实执行次数；shapes_canary.json冻结形状与canary；quality_bootstrap.json新增相对Exact区间。", "",
        "**本轮决定：保留冻结完整native设计，当前没有由FLOPs要求立即简化的理由。** 基础质量收益与条件规模算术优势可以并列；native及源条件的AUC必要性、百万用户质量泛化、持续迁移/短历史和实际时间节省仍未成立。无新结构修改、无校准扩展、无确认读取；本轮在账本交付后收束。", "",
        "计数参考：[NVIDIA矩阵乘法约定](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)采用FMA=2；[NVIDIA性能说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html)区分算术、带宽与实现性能。本报告的模型尺寸、算法与数值来自本地代码和冻结证据，不是这些网页给出的EvoKV结果。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n")


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    shapes_and_canary()
    raw=pd.read_parquet(RUN/"quality_raw.parquet")
    rows=population_counts(raw)
    cfg=json.loads((DESIGN/"native_base_ablation256_01/configuration.json").read_text())
    cohort=cfg["cohort"]; fit=cfg["fitting_uids"]; old=cohort["original_fitting_uids"]
    count_path=OUT/"calibration_counts.json"
    if count_path.exists() and "scene_validation" in json.loads(count_path.read_text())["C"]:
        counts=json.loads(count_path.read_text())
    else:
        print("Loading existing calibration timestamp metadata only",flush=True)
        history=histories(fit,CUTOVER_DAYS[-1]+1)
        counts=dict(C=calibration_counts(history,fit,set(cohort["lifetime_fitting_uids"])),
                    A=calibration_counts(history,old,set(cohort["lifetime_fitting_uids"])&set(old)))
        dump("calibration_counts.json",counts)
    result={}
    e=4*value(rebuild(1024))
    for method in METHODS:
        shared,solvers=shared_cost(counts["C"],counts["A"],method)
        pop=population_cost(rows,method)
        k=sum(shared.values()); a=sum(pop.values())/4091
        release=sum(v for name,v in pop.items() if name=="init" or name.startswith("publish"))/4091
        scales=[]
        for u in (4091,30000,100000,1000000):
            scales.append(dict(users=u,exact_TF=u*e/1e12,method_TF=(k+u*a)/1e12,
                release_ratio=(k+u*release)/(u*e),life_ratio=(k+u*a)/(u*e),saving=1-(k+u*a)/(u*e)))
        thresholds={}
        for label,beta,cost in (("life_break_even",1,a),("release_20pct",.2,release),("life_20pct_optional",.2,a)):
            thresholds[label]=k/(beta*e-cost) if beta*e>cost else None
        q=sum(r["requests"] for r in rows)/4091
        perquery=pop["read"]/sum(r["requests"] for r in rows)
        # Higher scoring scenarios conservatively add one new dirty refresh per added
        # scoring call, capped by real writes; preserve event rate. Not production evidence.
        events=sum(r["appends"] for r in rows)/4091
        refresh=sum(r["refreshes"] for r in rows)/4091
        # A true bound uses the most expensive target and two retained segments,
        # not the observed mean cost, whose target mix could change with traffic.
        bound=prepare_cost(5,2,method=="no_source")
        boundkeys=("pack","features","pca","view")+(("means",) if method=="summary_input" else ())
        unitrefresh=sum(bound[key] for key in boundkeys)
        sensitivity=[]
        for factor in (1,10,100,1000):
            added=q*(factor-1)
            afixed=a+added*perquery
            abound=afixed+min(added,max(0,events-refresh))*unitrefresh
            sensitivity.append(dict(query_multiplier=factor,queries_per_user=q*factor,
                life_ratio_30k_same_refresh=(k/30000+afixed)/e,
                life_ratio_30k_extra_dirty_bound=(k/30000+abound)/e))
        result[method]=dict(shared_FLOPs=shared,solver_FLOPs=solvers,population_FLOPs=pop,K=k,e=e,a=a,release_per_user=release,
            scales=scales,thresholds=thresholds,query_sensitivity=sensitivity,
            asymptotic_life_ratio=a/e,
            extra_queries_to_break_even_30k_same_refresh=max(0,(e-k/30000-a)/perquery))
        # Candidate ledger: algebraic normalization folding, same learned function.
        # no_source has global b/A so source-dependent state and its preparation vanish.
        # This is a candidate mathematical schedule, NOT measured speed or new quality.
        global_fold=4*L*(D*D+2*D*D+3*(R+1)*H*(d+1)*d)
        candidate=dict(pop)
        if method=="no_source":
            candidate={"read":sum(r["requests"] for r in rows)*(2*L*D*D+2*L*H*d*d+5*L*D)}
            candidate["publish_global_view"]=4*sum(prepare_cost(1,1,True)[key] for key in ("view",))
        else:
            for label in ("publish","refresh"):
                cnt=sum(r["publications"] if label=="publish" else r["refreshes"] for r in rows)
                candidate[label+"_view"]=cnt*(2*L*H*(R+1)*(d+1)*d+L*H*(d+1)*d)
            candidate["read"]=sum(r["requests"] for r in rows)*(2*L*D*D+2*L*H*d*d+4*L*D)
            if method=="summary_input":
                candidate["read"]+=sum(r["requests"]*L*((4*D+4*H)*(r["target"]+1)) for r in rows)
        ca=sum(candidate.values())/4091
        result[method]["candidate_equivalent"]=dict(population_FLOPs=candidate,K=k+global_fold,a=ca,
            life_ratio_30k=((k+global_fold)/30000+ca)/e,
            scope="fixed learned coefficients, normalization folded; global no-source view; not deployed or wall-clock validated")
        numerical=[]
        for sp,eg in ((1,9),(8,20),(32,20)):
            sk,_=shared_cost(counts["C"],counts["A"],method,sp,eg)
            se=4*value(rebuild(1024),sp)
            ma=a+(sp-1)*sum(r["requests"]*L*H*(r["target"]+1) for r in rows)/4091 if method=="summary_input" else a
            numerical.append(dict(special_function_weight=sp,eigen_cubic_allowance=eg,K=sum(sk.values()),e=se,
                                  life_ratio_30k=(sum(sk.values())/30000+ma)/se))
        result[method]["numerical_convention_sensitivity"]=numerical
        # Hypothetical length sensitivity, fixed event/query rate and fixed calibration
        # from current1024 design; no new quality evidence at these lengths.
        length=[]
        for n in (16,128,512,1024):
            newinit=4*V*(n-1)
            na=a-pop["init"]/4091+newinit
            ne=4*value(rebuild(n))
            length.append(dict(history_length=n,life_ratio_30k=(k/30000+na)/ne,
                               scope="fixed K and service rates, one-slot arithmetic; counterfactual cost only"))
        result[method]["length_sensitivity"]=length
    dump("ledger.json",result)
    # Existing feedback only, original globally paired UID resampling; add Exact contrasts.
    protocol=json.loads((DESIGN/"native_base_protocol_01/configuration.json").read_text())
    boot=bootstrap(raw,protocol["development_uids"],1000)
    dump("quality_bootstrap.json",boot)
    files=[RUN/"quality_raw.parquet",RUN/"state_counts.json",DESIGN/"native_base_protocol_01/configuration.json",
           DESIGN/"native_base_ablation256_01/configuration.json",__file__]
    dump("provenance.json",dict(inputs={str(p):hashlib.sha256(open(p,"rb").read()).hexdigest() for p in files},
        frozen_weights_sha256=json.loads((RUN/"shard_00000/configuration.json").read_text())["weights_sha256"],
        confirmation_read=False,new_model_fits=0,new_population_forwards=0))
    render(result,rows,raw,boot)
    print(json.dumps({m:{"K_TF":v["K"]/1e12,"e_GF":v["e"]/1e9,"a_GF":v["a"]/1e9,"scales":v["scales"],"thresholds":v["thresholds"]} for m,v in result.items()},indent=2),flush=True)


if __name__=="__main__":
    main()
