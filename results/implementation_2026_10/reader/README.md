# 融合动态读取验证

冻结选择 `A_itemgate6_rms_roll4` 的读取函数保持不变。Q≤4采用每Query/head一个CTA的FP32元素乘与规约路径，没有Query tile补齐、不使用MMA；Q>4采用Query16/历史64/码宽16的矩阵路径。两条路径均直接从48维共享/私有码寻址当前共享3＋私有3，在ELU+1之前合成score纠正，同一个历史tile内累计原生V和低维Z，再执行Value投影。不生成按头展开的全历史码、score/weight矩阵或纠正K/V副本。零码后缀只跳码运算，原生读取完整执行。

权威结果为`report.json`，命令和过程见`index.json`。558个合成case覆盖N0/1/17/63/64/65/127/513/1024、Q1/2/4/5/16/32、零/短/半/满prefix、因果FIFO/全遮挡/数值批mask和非连续FIFO视图；未使用码坐标置10000不改变结果。13项pytest通过。8个已有冻结校准来源用户各测Q1/4/16，共24次操作、144层读取，用于数值兼容检查，没有拟合、未来标签或新质量人口实验。

合成最差相对L2为6.98e-7、绝对差2.60e-4。实际读取最差相对L2：向量1.24e-7，矩阵6.67e-7；两者最终logit最大差均4.77e-7。原始ELU+1输出峰值可达1e7，raw差可为数个单位，不能声称bitwise等价。首次矩阵核固定raw绝对0.01阈值失败，保留`probe_attempt1.log`及`attempt1_failure.json`。最坏归一单层另以FP64 history算术为oracle：Torch相对L2 7.53e-8，矩阵核5.92e-7，数值精度略差，这一不利结果保留。

N1024/Q16/满prefix的live PyTorch CUDA tensor增量峰值：参考2,125,824B，融合29,184B。均含输出，排除输入、寄存器/共享内存及CUDA reserved memory。这不是加载服务内存峰值或DRAM流量。融合不减少有用逻辑FLOPs；`padding_and_backend_review.json`区分逻辑算术、几何padding、精度lowering和请求字节。mask模型默认bool1B；浮点mask作语义压力检查，流量须乘其元素字节数。

**精度与padding。** FP32指输入、输出、状态及累加。向量核采用FP32乘法/规约、历史tile64、码宽补齐8；矩阵核所有`tl.dot`明确使用`input_precision='tf32x3'`补偿式TF32乘法，码宽补齐16。Torch关闭`allow_tf32`不会改变Triton矩阵乘法。N1024/满prefix，Q1当前几何工作约为有用工作的1.053倍，Q16为1.267倍；矩阵TF32x3另涉及多MMA pass，因此几何数不是实际硬件FLOPs。旧全Query矩阵核Q1曾有20.27倍几何浪费，旧源码和报告已归档`matrix_only_prior/`，现由Q≤4向量核取代。未测生产吞吐/p99，不把这些模型数当计时收益或原20%预算进一步下降的证明。

接口`reader.py:fused_relay_read`支持显式`backend='triton'/'torch'`，错误不回退。`prepared_gates=(gK,gV)`各为`[B,H,Q,6]`；`prepared_decoders=(dK,dV)`各为连续`[H,6,32]`，供发布级参数缓存与每操作全层门控复用。限定H6/D32、共享3/私有3、legacy ELU+1、无相对位置偏置/训练dropout、无梯度推理；训练继续原可微实现。
