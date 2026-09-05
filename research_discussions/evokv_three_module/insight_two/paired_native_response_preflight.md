# Paired native-response preflight：完整 activation 后的 trajectory control variate

日期：2026-09-03  
状态：**NO-GO / RETIRE；固定 UID1930、五 edge、held-out odd-32 的非正式 route elimination**

## 1. 裁决

本轮检验 paired r4/r4 replay 是否只是被 K/V splice 或 P8 affine compiler 放错了边界。主方法不使用
`U0`、probe、moment、mapper 或 Current Exact upper state，而让 Current query 在每层执行：

\[
R_l^{mig}(q)=R_l^C(q;K_l^P,V_l^P)
 +R_l^C(q;\widehat K_l^C,\widehat V_l^C)
 -R_l^C(q;\widehat K_l^P,\widehat V_l^P).
\]

两条近似轨迹各自保留自己的 token factors；差分发生在原生 query--key activation 与 value
aggregation 之后，再进入原生 output projection、gate 和 residual。这个构造比 common-projection
control 更严格地保留 paired trajectory，也比 P8/S4 compiler 少一个 activation-region 近似。

结果为：

```text
paired-r4 native response  .8725/.9214/.9549/.9335/.8236, mean=.9012
single-current-r8 cache    .9134/.9986/.8515/.9814/.9372, mean=.9364
single-current-r8 U0       .8610/.9173/.9852/.9473/.9753, mean=.9372
```

主方法只在 `1/5` edge 胜 single-r8 reduced cache，只在 `2/5` edge 胜 single-r8 shared-`U0`；均值分别
低 `3.53pp` 与 `3.61pp`。它的 constructor 还略贵于 single-r8。因此按运行前硬门直接
**NO-GO / RETIRE**，不扩用户、不做 formal canary，也不把“native response control variate”写成
Design 1。

## 2. 同 runner 控制

所有合法状态均在 materialize Current Exact 前完成。只读 frozen discovery 的第一个 UID `1930`，评价
只用 held-out odd-32；P8 只供旧 S4 control 使用，主方法不读 probe。没有 rank、seed、layer 或
candidate sweep。

| method | v0→v1 | v1→v2 | v2→v3 | v3→v4 | v4→v5 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paired r4/r4 native response | .8725 | .9214 | .9549 | .9335 | .8236 | **.9012** |
| paired r4/r4 K/V splice | .8703 | .9015 | .9849 | .8685 | .7059 | .8662 |
| paired r4/r4 P8 S4 | .8725 | .9212 | .9487 | .9335 | .8226 | .8997 |
| common-projection r8 native response | .2722 | .9337 | .8316 | .9878 | .9362 | .7923 |
| single Current-r8 reduced cache | .9134 | .9986 | .8515 | .9814 | .9372 | **.9364** |
| single Current-r8 shared-`U0` | .8610 | .9173 | .9852 | .9473 | .9753 | **.9372** |

相对 paired K/V splice，native response 在 `4/5` edge 提高，mean 增加 `3.49pp`。这说明 nonlinear
reader 后做差比先合成一份 K/V 有真实作用；但它没有产生 paired-specific advantage：相同总 rank 的
generic single Current replay 仍更好、更便宜。native 与 P8/S4 的均值只差 `0.15pp`，也说明当前主要
误差来自两条 r4 trajectory，而不是 P8 activation-region compiler。

## 3. Exact-limit 与因果检查

实现冻结了以下代数门：

1. `rank = full token rank`、exact SVD 时，paired recurrence 恢复两条 exact release trajectory；
   `Parent exact + Current response - Parent response` 恢复 Exact Current reader。
2. `Parent = Current` 时，即使低 rank，两个 paired factors 完全相同，signed response 为零，输出等于
   Reuse/Exact。
3. eviction 对 factor state 闭合：对 exact Parent K/V 与两臂 `L` 删除相同历史行即可，`C_K/C_V`
   不变，因为 row selection 与 `L C` 严格交换。
4. 已有合法 Current suffix 时，legacy HSTU 的未归一化 pointwise response 可按不相交 token segment
   相加；cutover sidecar 无需重写。测试把 full-rank cutover factors 与 exact Current suffix 组合，恢复
   完整 Current cache reader。

第 4 条只证明 **read composition**。生产 append 必须让每个新事件通过同一 corrected reader 形成其
逐层 Current K/V，再写入 Current suffix；不能把普通 stale-Parent append 当成已经证明的 constructor。
本轮也不把未归一化 HSTU 的 segment additivity 外推到 softmax reader。

## 4. 成本、sidecar 与请求 I/O

严格从已审计 matrix-free paired final-K/V ledger 起算：

```text
paired final-KV ledger                    874,402,376
- superseded shared-U0 builder              1,247,808
- superseded signed K/V core build             916,480
= paired native-response constructor       872,238,088
= 18.2810% Exact-All                         PASS <20%
```

Medium `L=6,N=1024,Hkv=192,r=4` 的持久化 sidecar 保存两臂、每层各一份
`(L_token, C_K, C_V)`：

```text
2 * 6 * (1024*4 + 2*4*192)
= 67,584 scalars
= 270,336 bytes at FP32
= 2.8646% of one complete Parent K/V cache
```

每个 query、每层必须读取 Current 与 Parent 两份 factor，即每层两次、六层共十二次 factor read。按已有
FLOP 口径，每份 r4 native factor response 为 `101,376` FLOPs；加两次 H-wide signed add 后，增量为
`1,218,816 FLOPs/query`。另有 `73,728` 次 native activation evaluation/query，并单列
`67,584` 个 logical sidecar scalar reads/query；具体 kernel 若重复读取 `L`，必须在 runtime roofline
中按实测 bytes 重新报告。

这与 single-r8 的 `853,836,992 = 17.8953%` constructor 相比没有成本优势。它的 reader arithmetic
与 shared-`U0` 的 `1,254,528 FLOPs/query` 接近，但 sidecar 从 `26,624` 增至 `67,584` scalars。

## 5. 科学含义

本轮允许一个克制的负面观察：

> 把 matched Parent/Current trajectory 的差分后移到完整 native attention response，能够过滤一部分
> K/V-space 合成误差；但 response boundary 本身不会创造低 rank paired replay 中缺失的 Current
> information。

因此失败不能再归因于 shared-`U0` 或 P8 moments。两臂 control variate 在 full-rank 极限语义正确，
但 r4/r4 时并未比同计算量的 single-r8 更准确。这条路线依然是 paired low-rank replay 加标准
control-variate placement；没有通过“Parent-specific component 带来 generic compression 没有的稳定
收益”这一创新门。

后续不应围绕此路径扫 rank、basis、probe 或 layer。若继续寻找论文级 Design，需要新的 Current
information source 或 migration-ready state interface，而不是继续改变同一 reduced trajectory 的
subtraction 位置。

## 6. 实现与验证

- `scripts/insight_two/paired_native_response.py`：two-arm factor sidecar、native reader、row eviction、
  Current-suffix composition 与 Medium cost ledger；
- `scripts/insight_two/run_paired_native_response_preflight.py`：固定 UID1930/five-edge/all-controls runner；
- `tests/test_insight_two_paired_native_response.py`：full-rank exact、zero-defect、eviction、suffix composition
  与成本测试。

Focused verification：`5 passed`；相关文件通过 `ruff check` 与 `py_compile`。runner 未写 formal result、
contract 或 seal，未读取 confirmation users 或 labels。
