# 旧 ref_soft 在当前数据划分上的复验

用户于 2026-09-10 授权执行。先覆盖原先报告正增长的 WSODD 和 FloW；当前 SeaDronesSee 的五类、source-disjoint 数据留作后续外部检验，本队列不扩展到它。

## 固定比较

| 组 | 模型与损失 | box / dfl | 离线追加 |
|---|---|---|---|
| A baseline | YOLOv8s + CIoU | 7.5 / 1.5 | 无 |
| B weights_only | YOLOv8s + CIoU | 4.0 / 1.3 | 无 |
| C bundle | 历史 P3/P4 C2f_CBAM + 历史 Focal-EIoU | 4.0 / 1.3 | 无 |
| D duplicate | 同 C | 4.0 / 1.3 | 原样本的 10% 干净副本，同编码方式 |
| E refsoft | 同 C | 4.0 / 1.3 | 原样本的 10% 历史 ref_soft 副本 |

预先固定 seed 42、0、1，每个数据集五组，共最多 30 个完整训练；先跑 seed 42，再补 0/1。不根据 test 结果筛选组别或种子。若数值异常、输入变动或磁盘不足，保留证据并停止队列。早停沿用旧 patience=100，最高 150 epochs。

B−A 是损失权重改变；C−B 是 CBAM 与 Focal-EIoU 联合改变；D−C 是追加样本所带来的重复采样、重编码、额外训练步数的合并效果；E−D 才是数量、选图和编码匹配后的反光变换效果。E−C 是旧增强方案的总效应，E−A 是旧论文式整套方案总差。C−B 不单独归因于 CBAM 或 Focal-EIoU；D−C 不单独归因于步数。该设计不估计反光与模型结构的完整交互作用。

## 原样保留与必要适配

复制本地留存的 Ultralytics 8.3.182 自定义源码，逐文件 SHA-256 锁定，进程独立导入，不修改 AutoDL 当前安装包。CBAM、Focal-EIoU、DFL、assigner、模型 YAML、光照函数、随机数抽样及保存编码均沿用留存实现。旧 baseline 与改进环境的 trainer、标准增强、CBAM primitive 和验证 metrics 文件已经逐字节比较相同；CIoU 路径还要用历史 baseline loss 做数值及梯度对照。

从 YAML 随机初始化，不加载预训练。训练依历史实际 args.yaml：imgsz=960，batch=4，SGD，lr0=0.006，lrf=0.01，momentum=0.937，weight_decay=0.0000625，nbs=8，150 epochs，AMP，workers=8，warmup=3，mosaic=0.7，mixup=0.05，close_mosaic=10，hsv=(0.015,0.6,0.2)，scale=0.45。其它缺省项也继承旧落盘参数；仅路径、输出名、设备、种子和非训练绘图/日志适配。保存解析后的全量 args 与运行中的 criterion、结构、参数更新、初始化指纹。

反光：sample_ratio=0.10，low_light_prob=0，reflection_prob=1，allow_combined=false，max_area_ratio=0.035，min_area_factor=0.35，aspect=1.4..3.2，intensity=16..34。调用原 builder，因此保留 choose_transform 消耗随机数的顺序。选图及图像随机数固定 seed=42，三个训练种子共用相同覆盖集，检验训练随机性；不宣称覆盖增强种子的变异。干净副本对相同 ID 使用 PIL RGB 读取及 quality=95 保存。保持同名后缀 `_rf`，使两组排序、样本数量、DataLoader 顺序对应；标签逐字节一致。约 10% 追加意味着副本占扩增后数据约 9.1%。

数据替换为当前 wsodd_sequence_disjoint_v1（3977/1084/2406）和 flow_source_temporal_disjoint_v1（653/203/344），不复用旧随机图像划分。train/val/test 清单、标签和图像内容摘要冻结，增强仅来自 train。修正原 builder 输出 YAML 中相对 test 路径的搬迁问题，实际评估始终读取固定原始 split 清单。Linux 使用 AutoDL 现有 torch/CUDA/numpy/Pillow 等二进制依赖，记录差异；不声称逐比特复现旧 Windows 环境。当前 torch 2.5.1、numpy 2.2.6 与旧记录一致，Pillow 版本变化明确保留为环境边界。

## 验证和评估

启动前：源码哈希；选图、标签、编码与原函数一致；无 train/val/test ID 或相同图像内容交叉；CIoU/Focal-EIoU 前向与梯度等价；五种训练配置在真实图片上完成短训练，确认 scratch、模型 s scale、P3/P4 插入位置、criterion 和优化器实际更新。

每轮统一原 val。历史 trainer 的 best.pt 按 0.1*mAP50+0.9*mAP50-95 选；原报告展示的峰值 mAP50-95 可能是另一个 epoch，因此额外保存 best_map.pt，明确分别汇报，不混称同一检查点。主 test 比较使用历史 best.pt；best_map.pt 是预先声明的检查点选择敏感性检查。各组评估参数固定，使用现有 COCO area 评估器记录 mAP50-95、mAP50、APsmall、分大小及分类结果。测试只在所有训练条件预先冻结后评价，不进入训练或选组。

最终报告包含每个种子、配对差值、均值/样本标准差、正负次数、训练峰值 val 与固定 test 的差异、训练轮数/优化器步数。三个种子只能提供有限稳定性证据；不以一次正增长或峰值选择证明普适收益。若 E−D 无一致收益，则不把 E−C 的改善归因于反光。若当前划分不能重现旧增长，只能否定其在当前协议下的复现，仍不能单凭本实验把原因唯一确定为旧划分或 Windows/AutoDL 环境。

所有旧结果保留，不提交或推送 Git。输出独立目录 `/root/autodl-tmp/refsoft_reproduction_20260910/release_v1`。队列只启动一次，并用文件锁排除重复启动。
