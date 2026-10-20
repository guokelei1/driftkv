#!/usr/bin/env python3
"""CPU-only paired UID uncertainty and concise report for the fixed V4 probe."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main(args):
    config=json.loads((args.output/"configuration.json").read_text())["config"]
    quality=json.loads((args.output/"quality.json").read_text())
    rows=pq.read_table(args.output/"scores.parquet").to_pylist()
    uncertainty=dict(repetitions=1000,seed=20261019,method="Paired resampling of full fixed UID panel, pooled AUC, percentile95; same resampled UID counts for every method. Single-class replicates omitted.",panels={})
    fields=("fit16","fit32","fit64","reuse","rebuild")
    for name,users in (("registered32",config["evaluation_uids"]),("fixed_first128",config["trend_evaluation_uids"])):
        uncertainty["panels"][name]={}
        lookup={u:i for i,u in enumerate(users)}
        for protocol in config["protocols"]:
            group=[r for r in rows if r["uid"] in lookup and r["protocol"]==protocol]
            uid=np.array([lookup[r["uid"]] for r in group]);labels=np.array([r["label"] for r in group])
            pos,neg=labels==1,labels==0
            indices={}
            for field in fields:
                _,index=np.unique([r[field] for r in group],return_inverse=True)
                indices[field]=(index,int(index.max())+1)
            def aucs(counts):
                row_weight=counts[uid];p,n=row_weight[pos].sum(),row_weight[neg].sum()
                if p==0 or n==0:return None
                result={}
                for field,(index,size) in indices.items():
                    wp=np.bincount(index[pos],weights=row_weight[pos],minlength=size)
                    wn=np.bincount(index[neg],weights=row_weight[neg],minlength=size)
                    result[field]=float(np.sum(wp*(np.cumsum(wn)-.5*wn))/(p*n))
                return result
            observed=aucs(np.ones(len(users),dtype=int))
            for field,value in observed.items():
                assert abs(value-quality["panels"][name][protocol]["auc"][field])<1e-12
            rng=np.random.default_rng(20261019)
            contrasts={"fit16_minus_fit64":[],"fit32_minus_fit64":[],"fit64_minus_reuse":[],"rebuild_minus_reuse":[]}
            for _ in range(1000):
                values=aucs(np.bincount(rng.integers(0,len(users),size=len(users)),minlength=len(users)))
                if values is None:continue
                for contrast,samples in contrasts.items():
                    left,right=contrast.split("_minus_");samples.append(values[left]-values[right])
            uncertainty["panels"][name][protocol]={key:dict(valid_replicates=len(value),
                uid95=np.quantile(value,[.025,.975]).tolist()) for key,value in contrasts.items()}
    (args.output/"uncertainty.json").write_text(json.dumps(uncertainty,indent=2)+'\n')
    train=json.loads((args.output.parent/"training_summary.json").read_text())
    lines=['# V4 增量校准人数：固定配方的小规模证据','',
        '固定 V3 方法、source 与所有训练目标，只把 V4 的拟合用户改为原有序 64 人中的前 16/32 人；保留同一 16 人验证集、local8/joint4、batch8、seed17 和短尾4步。原64权重与分数复用。固定原登记32人与原4096顺序前128人，共159人的并集一次执行，两组分开报告；无人按结果挑选。','',
        '|V4 拟合人数|记录的校准流程秒数|同16人验证 teacher MSE|V4校准 FLOPs（T）|','|---|---:|---:|---:|']
    for row in train['records']:
        lines.append(f"|{row['fit_users']}|{row['measured_elapsed_seconds']:.2f}|{row['validation_teacher_mse']:.5f}|{row['calibration_flops']/1e12:.3f}|")
    lines+=['','16/32 是本轮 V4 增量拟合人数。三个分支都继承已经训练的256人起点与64人 V3 模型，不能写成“整个方法只需16人”。原64计时来自历史执行，非同期受控速度比较；完整80人的 source/teacher 构造仍是固定保留费用，未按人数除小。','',
        '|固定开发面板 / 协议|Reuse AUC|Rebuild AUC|fit16|fit32|fit64|候选 / 负例|','|---|---:|---:|---:|---:|---:|---:|']
    for panel in ('fixed_first128','registered32'):
        for protocol,record in quality['panels'][panel].items():
            a=record['auc'];label='next-item16' if protocol.startswith('behavior') else '原始feedback'
            lines.append(f"|{panel} / {label}|{a['reuse']:.5f}|{a['rebuild']:.5f}|{a['fit16']:.5f}|{a['fit32']:.5f}|{a['fit64']:.5f}|{record['candidate_scores']} / {record['candidate_scores']-record['positive_labels']}|")
    lines+=['','固定128人上，两协议点估计均随16→32→64改善；不能据此说16/32已经足够，也没有测到更大人数是否继续改善。Next-item 的重建增益分母很小，恢复百分比会超过100%，正文宜比较直接 AUC 差。32人 feedback 仅8个负例且 Rebuild 低于 Reuse，不适合判断恢复率或替128人趋势作结论。','',
        '|固定128人：与64人 AUC 差|点估计|成对 UID 95%区间|','|---|---:|---|']
    for protocol,r in quality['panels']['fixed_first128'].items():
        label='next-item16' if protocol.startswith('behavior') else 'feedback'
        for count in (16,32):
            ci=uncertainty['panels']['fixed_first128'][protocol][f'fit{count}_minus_fit64']['uid95']
            lines.append(f"|{label} fit{count}−64|{r['auc_delta_to_fit64'][f'fit{count}']:.5f}|[{ci[0]:.5f}, {ci[1]:.5f}]|")
    lines+=['','固定 epochs 与 batch 后，人数增多也增加优化步数。这是原训练配方的“人数—成本—质量”敏感性，不能把收益完全归因于样本多样性；纯样本效率需要等步数控制。所有结论只覆盖固定前版下的一条 V4 边，不证明跨版本全链、其他模型规模或工业人口所需拟合量。','',
        '全量159人/两协议共38,864条候选评分，实际评估196.87秒。原64在每协议前4个固定用户的重评分最大绝对误差4.77e-7，其他64分数按完整候选身份复用。最初评估在开始评分前因浅复制只读代理失败；已保留失败记录并改为共享只读对象、复制可变会话状态，未改训练或数学逻辑。','',
        '详细质量与区间见 quality.json、uncertainty.json；原分数和配置见同目录 scores.parquet、configuration.json；训练账见 ../training_summary.json。此轮不选择新最佳参数，不重跑原64训练，不扩展校准角色。']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(uncertainty['panels']['fixed_first128'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
