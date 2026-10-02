# v2 拟合脚本目录整理前的源文件

2026-09-28 的 `initial` / `batched` 开发探针保留各自实际执行时的源文件哈希。
在两个 batched H 探针正常结束后，保存了当时的这两份源文件原文，再执行方法目录隔离。
既有探针结果和哈希没有修改。

| 原路径 | 本目录副本 SHA256 | 当前实现位置 |
| --- | --- | --- |
| `scripts/read_correction_2026_09/v2/calibrate.py` | `2ae215ac78bcb36763f9e41a8a8eb00e6315f9db833233c757d895404b0a960b` | 共享准备/调度仍在原路径；Q 拟合移到 `v2/query_only/fit.py` |
| `scripts/read_correction_2026_09/v2/history_fit.py` | `8bad67621ae7723748f1c8bdebaa27787bcffbca21b6b1c931978a922c035b04` | `v2/history_conditioned/fit.py` |

这次整理没有改变数值实现：Q 的全部 8 个类/函数逐个通过忽略行号的 AST 相等检查；
H 文件与副本逐字节相同。更新了共享入口和已有测试的 import 路径。
归档的是整理前的这一时点；不将其声称为更早每个开发阶段的完整源代码快照。

后续正式 v2 运行按新目录重新记录源文件哈希；不会将布局前后的哈希直接混为一版。
