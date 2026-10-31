# E6 随机 SVD 逐矩阵能量证书草案

本草案仅新增 `scripts/ch3_motivation/energy_certificate.py` 与 CPU 数值测试；未导入或修改当前 GPU8 诊断的冻结源码。

令 Q 的列正交，B=QᵀM，E=(I−QQᵀ)M。由于 QᵀE=0，有 MᵀM=BᵀB+EᵀE。记 T=||M||F²，R=||B||F²，Cᵣ 为 B 的前 r 个奇异值平方之和，Sᵣ 为 M 的对应 exact 能量。投影收缩与 Ky Fan 不等式给出

`Cᵣ ≤ Sᵣ ≤ Cᵣ + ||E||F² = Cᵣ + T − R`。

因此 `Cᵣ / (Cᵣ + T − R)` 是当前 E6 所报告的 `captured_energy_ratio_to_exact` 的下界。R 必须在小矩阵 SVD 截到 rank16 之前，用全部奇异值计算；若误用截断能量，虽通常仍保守，却会把范围内的尾部也当残差，显著削弱证书。零矩阵单独采用比例为1的约定。

浮点版本不能只将负的 `T−R` 截为零后声称无误差证明。helper 在小能量摘要上用 float64 计算，并显式引入误差预算 a=`relative_energy_tolerance × T`。默认1e-5是本次 CPU fixture 的数值裕量，不能自动视为所有 GPU GEMM/SVD 的形式化误差上界；它假定 T、R 和每个 Cᵣ 的能量误差分别不超过 a，集成时应沿用代表矩阵 exact 诊断验证该裕量，保留假设。

若 δ 上界满足 `||QᵀQ−I||₂ ≤ δ < 1`，令 G=QᵀQ，则真实正交投影的 Gram 矩阵为 BᵀG⁻¹B。它的前 r 能量至多 `(Cᵣ+a)/(1−δ)`，总范围能量至少 `max(R−a,0)/(1+δ)`。因而 helper 使用

`residual_upper = max(T+a − max(R−a,0)/(1+δ), 0)`，

`lower_bound = max(Cᵣ−a,0) / ((Cᵣ+a)/(1−δ) + residual_upper)`。

这对“small SVD 奇异能量 / exact top-r 能量”保守，不是对下游 logit/AUC 误差的保证。δ 输入为零表示假定 Q 精确正交；若提供测得的 `||QᵀQ−I||F`，它本身上界谱范数，但还需包含 Gram 测量误差裕量。非有限值、无效 δ 或明显不一致能量均认证失败，不自动通过。近零但非零的矩阵不按固定绝对阈值当零处理，保持尺度一致性。

接口 `sketch_energy_certificate(total_energy, all_sketch_singular_values, ranks=(2,6,16), relative_energy_tolerance=1e-5, orthogonality_defect=0., threshold=.98)` 接受批量摘要，返回范围能量、残差上界、逐 rank 下界、通过标志与 `needs_refinement`。张量留在原设备，内部没有逐 head 的 CPU 同步。

最小集成位置是 `lowrank_batch.py` 的 small SVD 之后、`singular[..., :rank]` 截断之前；总 Frobenius 能量可沿用当前计算。只需保存完整谱的能量摘要及三个下界，无需长期保留 Q 或全 U。若标量分解路径仍可启用，必须同样接入，不能绕开守卫。将摘要并入已有一次批量 CPU 传输，避免逐矩阵同步。失败矩阵才增加 sketch 宽度/幂迭代或回退 exact，并记录实际采用的数值配置；当前冻结源结束前不集成。

该界可能很保守：要保证0.98，精确正交且忽略误差时需 `residual ≤ 0.020408 × Cᵣ`，所以 r=2 往往最难通过。平坦谱中随机子空间的 top-r 能量本来可以完全准确，但范围残差仍大，证书会拒绝；这类失败必须保留，不能反过来称算法失准，也不能承诺无需 exact。实际收益需由真实矩阵的逐 rank 通过率判断。

CPU 检查命令：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/ch3_motivation/test_energy_certificate.py`。结果11 passed，覆盖 float32/float64、1e-8/1/1e8尺度、近低秩、平坦谱、零矩阵、消减裕量、完整小谱与截断谱、非正交 Q 和无效输入；各有效 fixture 的下界均未高于 float64 exact oracle 的实际比例。

## GPU8 只读核对及后续精度门槛

`runs/e6/release_readiness_gpu8_edge2/attempt_1` 的8个用户分块已完成。四种variant、每用户10个代表head，共320个矩阵；r2/6/16产生960个exact能量比例检查，每variant/rank各80条。全局最小比例为0.9999387175（both_raw,r2），最大为1.0007771210（value_norm,r2）；全rank重建误差为0，120次serial/batched执行核对最大logit差为9.54e-7。实际exact检查耗时41.25秒，占这次81.40秒诊断的明显部分。

这些数值支持沿用0.98的经验精度门槛，同时说明代表head的通过不能代替所有未来用户矩阵核验。略大于1的比例反映有限精度或scalar/batch归约差异，不应解释为超过exact最优值。建议后续用全矩阵cheap guard选择需要加宽sketch/增加幂迭代/exact的矩阵，再保留代表exact检查与所有失败记录；记录下界通过率和实际回退费用。默认浮点裕量作为经验数值guard即可，不需要新增“必须证明全局浮点误差上界”的正式前提或批准环节，也不能把它表述为已经完成的形式化误差保证。此GPU8尚未保存未截断的small谱或Q，无法事后重建每个矩阵的cheap certificate通过率；需要最小集成后的独立小诊断核对。
