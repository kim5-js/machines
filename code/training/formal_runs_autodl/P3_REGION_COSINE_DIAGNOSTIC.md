# λ=0 checkpoint 的区域 P3 cosine diagnostic

本诊断只回答一个问题：clean/degraded 双视图之间的 P3 discrepancy 是否主要集中在目标区域，而不是背景。它不会训练模型、不会创建优化器，也不会写回 checkpoint。

## 预注册口径

- 固定使用正式 pretrained E2 λ=0 的 `best.pt`，并核对 SHA-256。
- 复用 λ=0 的 train dataloader、标准增强、batch=8、imgsz=960，以及锁定的在线 MDG。
- 模型使用训练模式前向以复现原 E2 统计，但所有参数 `requires_grad=False`，并在 `torch.inference_mode()` 下运行。
- global cosine 与既有实现一致：每幅图展平整个 P3 张量后计算。
- foreground 是增强后 GT 框投影到 P3 的网格并集；background 是其补集。
- small-object 是归一化框面积不超过 0.0025 的框并集，等价于 640 尺度下的 32×32。
- 正式诊断固定 200 batches；更短的运行只能作为 smoke，不具备 E3 放行资格。

## E3 放行条件

以下条件必须同时满足：

1. global cosine 与 λ=0 epoch 150 的 0.99601177 相差不超过 0.003；
2. 至少 500 个有效前景图像对；
3. 至少 70% batch 的 foreground discrepancy 大于 background；
4. 图像级配对差值 bootstrap 95% CI 下界大于 0；
5. foreground/background 的平均 discrepancy 比值至少为 1.05。

任何条件失败都停止在诊断阶段，不实现也不启动 E3。

## 输出

- `per_image.csv`：每幅图的 global/foreground/background/small-object cosine 和区域计数；
- `per_batch.jsonl`：每 batch 均值、MDG 计数和 foreground-background discrepancy 差；
- `diagnostic_summary.json`：描述统计、bootstrap CI、全部哈希与 E3 门槛结论；
- `diagnostic_protocol.json`、`events.jsonl`：预注册配置和执行事件。
