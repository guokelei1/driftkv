#!/usr/bin/env python3
"""Summarize the completed fixed4096 code-structure exploration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/code_structure_2026_10'
NAMES = {'group2': '每两 head 共享16', 'shared6_private7': '共享6＋私有7',
    'shared12_private6': '共享12＋私有6'}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    results = {l: read(OUT / l / 'quality4096/summary.json')['quality'] for l in NAMES}
    costs = {l: read(OUT / l / 'cost.json')[l] for l in NAMES}
    rows = ['# 共享紧凑码结构探索', '',
        '推荐后续继续研究“共享12＋每head私有6”：开发面板恢复89.4308%，完整新增FLOPs18.6369%，'
        '同总码存储下可见维数由8增至18。总体AUC点估计更高，配对区间仍跨零，尚不能宣称显著提升。', '',
        '固定 Medium HSTU（192维、6层、6 head、窗口1024），V2→V3→V4→V5连续回放。',
        '全部4096开发用户、51,731请求；同一256 fit /16 validation用户，local32复用后替换joint16。',
        '原模型、原生写入、首次读取初始化、FIFO和因果标记不变，所有候选最终参数在回放前固定。', '',
        '## 完整开发面板', '',
        '恢复率＝等发布平均候选–Reuse AUC增益／等发布平均 once-Rebuild–Reuse AUC增益。',
        '下表原方案也使用同样的静态4k生成优化；结构本身不领取仅属于实现加速的收益。', '',
        '| 结构 | 每head可见维数 | 恢复率 | 相对原方案平均AUC差（UID95%） | 完整新增FLOPs |',
        '| --- | ---: | ---: | --- | ---: |',
        '| 原head-local8 | 8 | 84.8066% | 参照 | 16.1120% |']
    compact = {}
    for layout, name in NAMES.items():
        q, cost = results[layout], costs[layout]
        p = q['overall']['paired']
        recovery = p['partial_rebuild_recovery_percent']
        delta = p['partial_minus_full_auc']
        ci = delta['percentile_95_interval']
        rows.append(f"| {name} | {cost['visible_floats_per_head']} | {recovery['estimate']:.4f}% | "
            f"{delta['estimate']:+.7f} [{ci[0]:+.7f}, {ci[1]:+.7f}] | {cost['complete_percent_of_once_rebuild']:.4f}% |")
        compact[layout] = dict(recovery=recovery, paired_auc_difference=delta,
            cost_percent=cost['complete_percent_of_once_rebuild'],
            source_reconstruction_percent=cost['complete_with_source_reconstruction_percent'],
            per_release={v: r['paired']['partial_rebuild_recovery_percent'] for v, r in q['by_release'].items()})
    rows.extend(['', '全部结构每层每历史行总共48个FP32数，等于原6×8；每用户码＋标记仍占原生K/V的12.5651%。',
        '这是开发探索，非盲确认。恢复率点估计超过80%并不证明稳定80%；UID区间及逐发布结果如下。', '',
        '| 结构 | V3恢复率 | V4恢复率 | V5恢复率 | 整体恢复率UID95% |',
        '| --- | ---: | ---: | ---: | --- |'])
    for l, name in NAMES.items():
        r = compact[l]
        values = [r['per_release'][v]['estimate'] for v in ['v3', 'v4', 'v5']]
        ci = r['recovery']['percentile_95_interval']
        rows.append(f"| {name} | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% | [{ci[0]:.3f}%, {ci[1]:.3f}%] |")
    rows.extend(['', '## 结构为什么具有更大表达空间', '',
        '原方案每个head独立把本地K/V和完整Current item embedding压成8维，'
        '该head的Key/Value联合残差只能位于至多8维输出子空间。',
        '共享结构在同一总存储内允许多个head访问一组码；共享分支读取覆盖head的原生K/V，'
        '私有分支保留本地信息，所有分支仍读取完整Current item embedding。'
        '共享6＋私有7允许每head使用13维；共享12＋私有6允许18维。',
        '读时将query投影至可见码空间，纠正Key分数，再在同一次历史注意力中聚合原生Value和码并解码。'
        '仍只有一份原生K/V，不保留纠正后的完整K/V。',
        '每个新结构严格包含旧函数：初始化时只开启原head自己的8个坐标，再通过原D2目标训练新增连接。'
        'CPU合成案例中，本地输入相同、另一head上下文不同，旧编码器无法区分，新结构可以。'
        '这证明表示类别扩大，不证明真实用户质量。'
        '已训练权重的跨head连接及解码器SVD记录在各版本expansion.json；V4有的候选仍选择零步。', '',
        '## 成本和边界', '',
        '完整账使用独立fixed30k、393,521真实服务请求，包含继承local32、新joint16、全部初始化/追加/读取、'
        '4worker各自的静态表准备。梯度范数、一致性比较及中心化坐标计算也已计费。'
        '源K/V是发布时已经存在的服务状态；额外重建它的敏感性单列。', '',
        '| 结构 | 原生完整query的额外开销 | 需重建旧K/V时的完整成本 | 共享张量／版本／worker |',
        '| --- | ---: | ---: | ---: |'])
    memory = read(OUT / 'memory_summary.json')['layouts']
    for l, name in NAMES.items():
        c = costs[l]
        m = memory[l]['by_release']['v3']['current_runtime_shared_tensor_payload_mib_per_worker']
        rows.append(f"| {name} | {c['extra_percent_of_native_whole_query']:.3f}% | "
            f"{c['complete_with_source_reconstruction_percent']:.4f}% | {m:.3f} MiB |")
    rows.extend(['', '查询开销增加，不能声称可忽略或已证明线上吞吐收益。共享张量包括adapter、'
        '校准坐标buffers、编码器参数副本和静态4k表；当前离线回放保留三发布对象，以上按一个发布计。',
        '16个校准验证用户上的连接消融未重新拟合，且他们参与过epoch选择：V5切除连接常使误差变大，'
        'V3方向不一致，V4零步候选不变。大量教师MSE仍来自共同分数偏移；MSE变化不替代AUC证据。', '',
        '共享12候选在同512用户的真实连续回放中，切除全部跨head编码/解码连接后，恢复率从98.6736%变为88.5254%。'
        '完整结构相对切除结构的平均AUC差为+0.0027246，UID95%为[-0.0004194, +0.0061420]。'
        '该干预支持新增连接参与了当前预测，但区间跨零；切除后的参数未重新优化，不能当成最优rank8对照。'
        '整体开发收益主要来自V4（69.3295%→79.8153%）；V3略有回退，不能声称各发布均改善。', '',
        '## 索引', '',
        '- 各结构 `quality4096/{summary.json,scores.parquet,final_checks.json}`：完整面板、UID区间和绑定检查。',
        '- 各结构 `cost.json`：完整组件、once-Rebuild分母、源重建敏感性和query开销。',
        '- 各结构 `calibration/v{3,4,5}/`：设置、权重、实际费用、连接/SVD诊断。',
        '- `structure_checks.json`、`serving_checks.json`、各分片`cost_shape_check.json`：函数包含、FP32等价、实际事件计数。',
        '- `frozen_candidates.json`、`memory_summary.json`、`logs/`：参数索引、共享存储和过程日志。',
        '- 共享6/12候选 `connection_ablation.json`；共享12 `ablation512_both/`：有边界的连接干预。', '',
        '512筛选与完整4096结果均保留，后者包含前者；不得把两个阶段称为独立重复验证。', ''])
    (OUT / 'report.md').write_text('\n'.join(rows))
    (OUT / 'summary.json').write_text(json.dumps(dict(status='complete', users=4096, requests=51731,
        candidates=compact, recommended_development_prototype='shared12_private6',
        recommendation_scope='Fixed-storage representation expansion, point recovery89.4308% and main complete cost18.6369%; no independent confirmation, improvement CI crosses zero, source reconstruction sensitivity22.1375%.',
        scope='Fixed-model development exploration, no independent confirmation'), indent=2) + '\n')
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
