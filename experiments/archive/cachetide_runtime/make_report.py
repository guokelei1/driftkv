#!/usr/bin/env python3
"""Join engineering evidence without altering any raw experiment result."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
from hstu_kvcache.common.artifacts import sha256, write_json

BASE=ROOT/'results/implementation_2026_10'
OLD=ROOT/'results/query_relay_2026_10'


def read(path):
    return json.loads(Path(path).read_text())


def check_sources(sources):
    for name,value in sources.items():
        path=Path(name)
        if not path.is_absolute():
            path=ROOT/path
        if sha256(path)!=value:
            raise RuntimeError(f'Execution source changed after evidence capture: {path}')


def build(*,allow_pending=False):
    reader=read(BASE/'reader/report.json')
    prepare=read(BASE/'prepare/final_full_catalogue/summary.json')
    state=read(BASE/'state/summary.json')
    cost=read(BASE/'cost/summary.json')
    integration=read(BASE/'integration/replay32_attempt1/summary.json')
    writing=read(BASE/'paper/writing.json')
    registration=read(BASE/'registration.json')
    freeze=read(OLD/'probes/A_itemgate6_rms_roll4/freeze.json')
    old_completion=read(OLD/'formal_aggregation/final_completion_evidence.json')
    old_feedback=old_completion['quality']['original_feedback']
    for name,value in freeze['source_sha256'].items():
        expected=(old_completion['checks']['cost_source_version_audit']['current_cost_source_sha256']
                  if name=='cost.py' else value)
        assert sha256(ROOT/'experiments/archive/query_relay'/name)==expected
    check_sources(reader['source_hashes'])
    check_sources(prepare['bindings']['source_sha256'])
    check_sources(integration['settings']['source_sha256'])
    check_sources({k:v for k,v in state['latest_files'].items() if k.endswith('state.py')})
    assert integration['status']=='complete' and len(integration['units'])==64
    assert integration['code_and_service_count_checks']
    assert len(registration['selected_uids'])==32
    assert {u['uid'] for u in integration['units']}==set(registration['selected_uids'])
    assert reader['status']=='complete' and reader['synthetic_cases']==558
    assert reader['all_finite'] and state['same_persistent_code_storage']
    for version in ('v3','v4','v5'):
        assert prepare['releases'][version]['full_catalogue']['P_Q_bitwise_equal']
        assert prepare['releases'][version]['full_catalogue']['work']['total_flops'] == \
            prepare['releases'][version]['full_catalogue']['original_compilation_flops']
        assert integration['settings']['calibration_weights'][version]['sha256'] == \
            freeze['release_bindings'][version]['calibration_sha256']
        assert integration['settings']['publications'][version]['old_artifact_sha256'] == \
            freeze['publication_bindings'][version]['sha256']
    tests=(BASE/'final_tests.log').read_text()
    passed=re.search(r'(\d+) passed in ([\d.]+)s',tests)
    if passed is None or int(passed[1])!=111:
        raise RuntimeError('Expected real final111-test log')
    paper=Path('/home/gkl/work/paper/new5.md')
    assert sha256(paper)==writing['target_sha256']
    assert sha256(Path('/home/gkl/work/paper/new4.md'))==old_completion['checks']['paper_design_sha256']
    assert paper.read_text().count('\n## ')==4
    assert all(word not in paper.read_text() for word in ('AUC','恢复率','111 passed','95%'))
    population_path=BASE/'population_confirmation1024/merged/summary.json'
    population=read(population_path) if population_path.exists() else None
    if population is None and not allow_pending:
        raise RuntimeError('Full1024 engineering replay is not complete')
    if population is not None:
        assert population['status']=='complete' and len(population['selected_uids'])==1024
        assert population['quality']['requests']==774096
        assert population['quality']['idle_uids']==22
        assert all(population['checks'].values())
        check_sources(population['source_sha256'])
    main=cost['protocols']['stride16_candidates16']
    feedback=cost['protocols']['original_feedback']
    assert main['twenty_percent_with_source_cost_ceiling_met']
    assert not feedback['twenty_percent_cost_ceiling_met']
    prepared_max=max(prepare['releases'][v]['full_catalogue']['D_difference']['max_abs']
                     for v in ('v3','v4','v5'))
    read_memory=next(row for row in reader['memory']['rows']
        if row['queries']==16 and row['history']==1024 and row['code_count']==1024)
    population_text='完整1024名已揭晓确认用户的工程复验正在两张空闲GPU上运行；完成前不报告新的质量结论。'
    if population is not None:
        q=population['quality']
        overall=q['overall']
        recovery=overall['paired']['partial_rebuild_recovery_percent']
        delta=population['optimized_minus_original_auc']['overall']
        population_text=(f"完整1024用户／774,096候选评分复验已完成，保留全部22名idle。优化方法等发布AUC "
            f"{overall['equal_release_mean_auc']['partial']:.10f}，相对原实现AUC差{delta['estimate']:+.10g}，"
            f"共同UID95%{delta['percentile_95_interval']}；最大logit差{population['numeric']['max_abs']:.10g}。"
            f"恢复率{recovery['estimate']:.4f}%，UID95%{recovery['percentile_95_interval']}。"
            '参考重建增益很小，比例不代表还原精确K/V；此处是原确认用户的工程复验，不是新的盲确认。')
    lines=[
        '# CacheTide Implementation：实现与工程验证',
        '',
        '**范围：当前冻结A/item/RMS方案的等价工程实现，不新增学习目标、不修改模型/校准权重或选择新的质量任务。** 四部分为共享码准备、融合读取、分块原地码转换、显式系统配置与集成。原`query_relay_2026_10`源码及实验资产保持原样。',
        '',
        '## 代码与技术稿',
        '',
        '`experiments/archive/cachetide_runtime/{prepare,reader,reader_kernel,state,backend}.py`提供实现；`integration.py`、`evaluate_population.py`与`population_analyze.py`提供冻结轨迹验证。`cost.py`仅作CPU理论账。论文 `/home/gkl/work/paper/new5.md` 已更新为四小节，仅写设计/实现；旧稿保存在 `paper/new5_before_implementation.md`。第四章未改，未编译论文。',
        '',
        '## 已完成组件证据',
        '',
        f"- **共享准备**：真实1,380,766ID、V3→V4→V5全目录重编译；P/Q逐位相同，D最大差{prepared_max:.3g}。每版lookup/物品投影调用510→85、embedding逻辑读取约6.36→1.06GB；这些是调用/逻辑载荷变化，投影FLOPs不变，也不是DRAM实测。固定φ的非线性位置和原生K/V项完整保留，φ/D各约1.59GB。权威结果`prepare/final_full_catalogue/summary.json`。",
        f"- **融合读取**：558个独立shape/mask/prefix组合、8个冻结真实源用户×Q1/4/16。N1024/Q16/满码前缀时，live CUDA tensor增量峰值{read_memory['backends']['torch']['peak_added_allocated_bytes']:,}B→{read_memory['backends']['triton']['peak_added_allocated_bytes']:,}B，包含输出而排除输入/on-chip空间。向量核覆盖Q≤4，矩阵核覆盖较大Q；未生成全历史score/weights或按头码副本。权威结果`reader/report.json`。",
        '- **原地状态**：同一codes storage完成反锚/物化、按块临时生成η；最终真实359行混合源、重复1024行及原Session三发布/FIFO均核对。历史32768行压力的转换临时峰值由参考144/180MiB变为chunk128约0.516/0.469MiB；该压力重复真实行，不作质量证据。该早期压力的wrap构造后来改为复用逆块，转换算术未变；当前源码由111项检查及完整轨迹另验。权威索引`state/summary.json`。',
        f"- **真实三版集成**：注册前仅按历史长度/72h活跃度选32名旧开发用户，四组各8人；两种协议×三条参考/优化路径完整回放64个用户协议单元、{integration['compared_candidate_scores']:,}候选评分。新读取、全部优化相对参考的最大评分差分别{integration['maxima']['reader_only_logit']:.3g}/{integration['maxima']['all_optimized_logit']:.3g}；事件、FIFO、出生零码、计数和账本相同。近零码按绝对＋相对混合容差检查。权威结果`integration/replay32_attempt1/summary.json`。",
        f"- **自动检查**：新旧相关检查111项通过，真实日志`final_tests.log`（{passed[2]}s，仅运行记录）。",
        '',
        '## 完整质量人口复验',
        '',population_text,
        '',
        '该组来自原Large-only确认面板，沿用Medium模型/映射与known-only候选；ID/标签/时间、原生控制、窗口/码长度及逐用户服务计数逐项匹配原结果。两片UID互斥，最终统一1024人共同UID重采样，不平均分片AUC或恢复百分比。部署成本仍由独立固定Medium30k账本计算，不以质量人口线性放大。',
        '',
        '## 完整理论成本与资源口径',
        '',
        f"主16/16行为任务完整新增FLOPs／一次K/V重建预算为 **{main['complete_percent_of_once_rebuild']:.5f}%**；包含完整校准来源重建为 **{main['complete_with_source_reconstruction_percent']:.5f}%**。只扣除了跨六层重复RMS（{main['saved_rms_flops']:,}F）；所有校准、共享表准备、纠正行为写入、查询、码转换及四副本独立准备继续收费。原保守93888F/行码转换口径保持。融合、投影合并及原地转换不再虚扣逻辑FLOPs。完整建模服务净省{main['net_saved_percent_of_total_full_service']:.5f}%，与20%重建预算分母分开。",
        '',
        '门控/decoder派生参数另占110,592B/副本；无新增逐行状态，逐行1152B码占9216B原生KV的12.5%。全局固定φ、D/P/Q/逆块及派生权重、新旧交接峰值单列，不能藏入逐行比例。Current门控的重复item lookup和普通服务流量不是零开销，主账本lookup41.760GB仍保留。见`cost/summary.json`与`cost/summary.md`。',
        '',
        '## 负例、数值与有效边界',
        '',
        f"- 原反馈完整4096质量边界仍为恢复{old_feedback['estimate']:.2f}%、80%非劣未获支持；本轮理论成本{feedback['complete_percent_of_once_rebuild']:.5f}%（含来源{feedback['complete_with_source_reconstruction_percent']:.5f}%），仍高于20%。本轮不通过工程优化声称解决原任务的质量失败。",
        '- 首个全Q矩阵核在Q1写入存在约20倍几何补齐，已归档`reader/matrix_only_prior/`；当前增加小Q向量路径。tile几何、TF32x3多pass、缓存命中和真实硬件执行量不能由逻辑FLOPs反推，见`reader/padding_and_backend_review.json`。',
        '- 原生ELU+1读取可达1e7，早期0.01绝对阈值失效保留；当前同时报raw绝对值、相对L2、FP64 oracle和最终评分。矩阵核对FP64的误差略大于Torch；FP32存储不等于IEEE FP32乘法或bitwise等价。',
        '- 持久码只有一份，追加/FIFO仍可能临时分配新张量；原地转换只保证转换工作区边界。当前同步原型未实现跨副本并发退休、故障恢复和加载生产服务吞吐/p99，不据组件计时外推。',
        '- 配置固定A更新器、H6/D32/read3+3、无相对位置偏置的ELU+1、无梯度推理。训练继续使用可微参考。有限三版与固定η/ID/模型结构仍是当前范围；原教师误差更大的边界不被排序质量替代。',
        '',
        '## 复现与索引',
        '',
        '`registration.json`固定32UID与容差；各组件README和执行日志保留命令/失败；`population_confirmation1024/index.json`记录互斥分片与聚合命令。本文及`completion_evidence.json`由`make_report.py`读取原始summary生成，完成状态要求完整1024聚合、111项检查和论文哈希一致。',
    ]
    (BASE/'report.md').write_text('\n'.join(lines)+'\n')
    required=['registration.json','reader/report.json','reader/padding_and_backend_review.json',
        'prepare/final_full_catalogue/summary.json','state/summary.json','state/gpu2_final.json',
        'cost/summary.json','cost/accounting.log','integration/replay32_attempt1/summary.json',
        'integration/replay32_attempt1/configuration.json','final_tests.log','paper/writing.json',
        'paper/new5_before_implementation.md','report.md']
    if population is not None:
        required += ['population_confirmation1024/merged/summary.json',
            'population_confirmation1024/merged/configuration.json',
            'population_confirmation1024/merged/scores.parquet']
    artifacts={name:dict(path=str(BASE/name),sha256=sha256(BASE/name)) for name in required}
    value=dict(status='complete' if population is not None else 'running',updated_utc=datetime.now(timezone.utc).isoformat(),
        scope='Four-section Implementation, unchanged frozen design/weights, component and continuous trajectory engineering verification',
        execution_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'experiments/archive/cachetide_runtime').glob('*.py'))},
        original_frozen_execution_sources_unchanged=True,
        original_source_comparison='Original Item freeze sources; cost.py compared with the already documented accounting-only version in the prior completion evidence',
        reader_cases=558,tests_passed=111,
        integration_units=64,integration_candidate_scores=integration['compared_candidate_scores'],
        integration_maxima=integration['maxima'],population_recheck_complete=population is not None,
        main_added_percent=main['complete_percent_of_once_rebuild'],
        main_added_with_source_percent=main['complete_with_source_reconstruction_percent'],
        original_feedback_added_percent=feedback['complete_percent_of_once_rebuild'],
        original_feedback_quality_boundary=old_feedback,
        paper=dict(path=str(paper),sha256=sha256(paper),sections=4,experimental_results=False,
            previous_backup=writing['backup'],design_unchanged_sha256=writing['design_sha256']),
        state_auxiliary_prepare_revision=dict(evidence_hash=state['latest_files'].get('src/hstu_kvcache/methods/cachetide/prepare.py'),
            current_hash=sha256(ROOT/'src/hstu_kvcache/methods/cachetide/prepare.py'),
            reason='Later metadata-compatibility guards do not change fixed_features_from_bank or state conversion arithmetic; current complete integration/population execute the new source'),
        pending=None if population is not None else 'Full revealed1024 engineering replay and joint bootstrap',artifacts=artifacts)
    write_json(BASE/'completion_evidence.json',value)
    write_json(BASE/'progress.json',{key:value[key] for key in ('status','updated_utc','pending','tests_passed',
        'reader_cases','integration_units','integration_candidate_scores','population_recheck_complete')})
    print(json.dumps(dict(status=value['status'],artifacts=len(artifacts),pending=value['pending'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-pending',action='store_true')
    build(allow_pending=p.parse_args().allow_pending)
