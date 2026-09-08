# 冻结decoder与闭环干预诊断

同场景、同面板；先场景平均再UID等权。所有数值为logit MSE，非AUC恢复率。

| 目标 | 路径 | 诊断held MSE | /Reuse | 中位数 | 优于Reuse的UID比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 | functional_response128/actual | 0.032588103 | 95.255 | 0.0140499 | 25.0% |
| M1 | functional_response128/free64 | 2.2153505e-06 | 0.0064755 | 2.88374e-07 | 100.0% |
| M1 | functional_response32/actual | 0.0032455621 | 9.4868 | 0.0036701 | 25.0% |
| M1 | functional_response32/free64 | 2.5194622e-06 | 0.0073644 | 7.67872e-07 | 100.0% |
| M1 | mean_coefficient32/actual | 0.00092377238 | 2.7002 | 0.000547994 | 25.0% |
| M1 | reuse/actual | 0.00034211289 | 1 | 0.000395464 | 0.0% |
| M1 | shared15/actual | 0.00065385211 | 1.9112 | 0.000498052 | 50.0% |
| M3 | functional_response128/actual | 0.0026631105 | 0.68066 | 0.00287019 | 25.0% |
| M3 | functional_response128/free64 | 4.4010208e-06 | 0.0011249 | 7.28781e-07 | 100.0% |
| M3 | functional_response32/actual | 0.0014616888 | 0.37359 | 0.000866372 | 75.0% |
| M3 | functional_response32/free64 | 7.5390473e-06 | 0.0019269 | 3.12293e-06 | 100.0% |
| M3 | mean_coefficient32/actual | 0.00068223853 | 0.17437 | 0.000718955 | 75.0% |
| M3 | reuse/actual | 0.0039125337 | 1 | 0.00229346 | 0.0% |
| M3 | shared15/actual | 0.00098651057 | 0.25214 | 0.000337887 | 75.0% |
| M4 | functional_response128/actual | 0.013153897 | 0.17175 | 0.00584699 | 75.0% |
| M4 | functional_response128/free64 | 5.9154434e-05 | 0.00077236 | 2.18439e-05 | 100.0% |
| M4 | functional_response32/actual | 0.0061865399 | 0.080775 | 0.00292506 | 100.0% |
| M4 | functional_response32/free64 | 5.5867436e-05 | 0.00072944 | 2.03652e-05 | 100.0% |
| M4 | mean_coefficient32/actual | 0.0065836619 | 0.08596 | 0.00495233 | 100.0% |
| M4 | reuse/actual | 0.076589635 | 1 | 0.0334682 | 0.0% |
| M4 | shared15/actual | 0.0031697739 | 0.041386 | 0.00228387 | 100.0% |
| M5 | functional_response128/actual | 0.029277897 | 0.65708 | 0.0156534 | 50.0% |
| M5 | functional_response128/free64 | 2.9691734e-05 | 0.00066637 | 8.63755e-06 | 100.0% |
| M5 | functional_response32/actual | 0.0083732462 | 0.18792 | 0.00159863 | 75.0% |
| M5 | functional_response32/free64 | 4.0555991e-05 | 0.0009102 | 2.14929e-05 | 100.0% |
| M5 | mean_coefficient32/actual | 0.022943551 | 0.51492 | 0.018718 | 50.0% |
| M5 | reuse/actual | 0.044557281 | 1 | 0.00731875 | 0.0% |
| M5 | shared15/actual | 0.018010958 | 0.40422 | 0.0032052 | 75.0% |

完整前缀/单层、拟合/留出、各UID权重和带符号交叉项见summary.json与decomposition.parquet。

确认集未读取；M2仅native过渡；M5 E14_partial。自由编码使用场景教师，非部署方法。
