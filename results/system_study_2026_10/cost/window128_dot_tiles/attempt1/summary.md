# 数学FLOPs与显式tile敏感性

- mathematical_effective: 新增/数学Rebuild=17.302169%；含来源=18.536790%。
- with_conversion_tile_sensitivity: 新增/数学Rebuild=20.681183%；含来源=21.915803%。
- with_conversion_and_reader_dot_tile_sensitivity: 新增/数学Rebuild=37.153280%；含来源=38.387900%。

后两项为已知method dot补齐敏感性，未计全部算子/机器指令；不称完整执行FLOPs。原生共同工作与Rebuild分母保持数学有效口径。
