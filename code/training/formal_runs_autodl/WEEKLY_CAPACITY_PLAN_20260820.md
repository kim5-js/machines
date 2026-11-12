# AutoDL RTX 4090 一周容量与费用计划

日期：2026-08-20

## 结论

- RTX 4090 24 GB 足够。统一 `imgsz=960`、`batch=8` 时，最重的
  RT-DETR-L 峰值预留显存约 12.52 GB，没有 OOM。
- 12 个现代 baseline（4 个模型 × 3 个数据集）按 150 epochs 线性外推
  约 62.37 GPU 小时；增加 15% 的保存和偶发重跑余量后约 72 小时。
- 推荐第一周只做 baseline 与 SeaDronesSee 的 E1-E4 顺序开发。该阶段
  含 20% 余量约 122.74-132.49 小时，可以放进 168 小时。
- 完整 32-run 论文矩阵含多种子和跨骨干实验，预计约
  222.51-262.18 GPU 小时（已含 15%-20% 余量），单卡约 9.3-10.9 天；
  不应承诺一周完成，按周租赁应准备两周。

## 计时口径

- 环境：Python 3.10.20、PyTorch 2.5.1+cu124、Ultralytics 8.4.124。
- 硬件：RTX 4090 24 GB，12 vCPU，90 GB RAM。
- 统一校准：`imgsz=960`、`batch=8`、`workers=8`，训练后执行完整验证。
- 下表为真实 AutoDL wall time；YOLO26 的 FloW/SeaDronesSee 是 2-epoch
  平均，其余是 1 epoch。校准关闭正式 checkpoint/plots，因此排程另加余量。
- `150 h = seconds_per_epoch × 150 / 3600`。这只是容量估算，不是论文结果。

| 模型 | 数据集 | 秒/epoch（含 val） | 150 epochs（小时） | 峰值预留显存（GiB） |
|---|---|---:|---:|---:|
| YOLOv8s | FloW | 25.453 | 1.06 | 4.39 |
| YOLOv8s | WSODD | 62.229 | 2.59 | 4.73 |
| YOLOv8s | SeaDronesSee | 162.346 | 6.76 | 4.75 |
| YOLO11s | FloW | 25.388 | 1.06 | 5.05 |
| YOLO11s | WSODD | 65.255 | 2.72 | 5.22 |
| YOLO11s | SeaDronesSee | 158.398 | 6.60 | 5.24 |
| YOLO26s | FloW | 27.241 | 1.14 | 6.25 |
| YOLO26s | WSODD | 82.827 | 3.45 | 6.51 |
| YOLO26s | SeaDronesSee | 174.558 | 7.27 | 7.29 |
| RT-DETR-L | FloW | 64.024 | 2.67 | 12.08 |
| RT-DETR-L | WSODD | 212.583 | 8.86 | 12.20 |
| RT-DETR-L | SeaDronesSee | 436.697 | 18.20 | 12.23 |

模型三数据集合计：YOLOv8s 10.42 h、YOLO11s 10.38 h、YOLO26s
11.86 h、RT-DETR-L 29.72 h。

## 组合后的 32-run 路线

两份项目文档按以下方式合并，避免重复实验：先用 YOLOv8s 在
SeaDronesSee 开发 E0-E4；方法成立后直接迁移到最终 MDC-YOLO26，
不再额外把 YOLOv8 E4 跑遍另外两个数据集。

| 阶段 | 唯一训练数 | 估算 GPU 小时 |
|---|---:|---:|
| 4 个现代 baseline × 3 数据集，seed 0 | 12 | 62.37 |
| SeaDronesSee E1-E4（E0 已在 baseline） | 4 | 39.91-48.03 |
| MDC-YOLO26 final，3 数据集，seed 0 | 3 | 18.98-23.72 |
| YOLO26 baseline 的 seed 1/2，3 数据集 | 6 | 23.72 |
| MDC-YOLO26 final 的 seed 1/2，3 数据集 | 6 | 37.95-47.44 |
| MDC-YOLO11 跨骨干确认（SeaDronesSee） | 1 | 10.56-13.20 |
| 合计（未加余量） | 32 | 193.49-218.48 |

未知的 MDC 训练开销采用显式假设：E1 的在线退化为 YOLOv8 baseline
的 1.1 倍；需要原图/退化图双路前向的一致性训练为对应 baseline 的
1.6-2.0 倍。实现完成后应再做 1-epoch 校准并替换此区间。

## 费用

按用户下单截图中的固定周价 `¥275.50 / 4090 / 周`：

| 方案 | 日历容量 | 费用 | 建议 |
|---|---:|---:|---|
| 1 × 4090，1 周 | 168 GPU h | ¥275.50 | 推荐：baseline + E1-E4 |
| 1 × 4090，2 周 | 336 GPU h | ¥551.00 | 完整矩阵最稳妥 |
| 2 × 4090，1 周 | 336 GPU h | ¥551.00 | 方法冻结后可并行多种子；开发阶段收益有限 |

等效周价约 `¥1.64/GPU h`，但周租通常按整周付费。按截图中的
下单前余额 ¥502.88 计算，若价格和余额都未变化，两周或双卡一周还差
¥48.12。

## 当前边界与风险

- FloW：当前只有官方 1200 张训练池的 1000/200 开发划分，缺少独立
  800 张 test；最终 test 结论仍被阻塞。
- WSODD：已生成 3977/1084/2406 的序列互斥 train/val/test。部分 JPEG
  会产生“EOI 前多 2 字节”的非致命 libjpeg 警告；在决定数据处理规则前
  不重编码原图。
- SeaDronesSee：已生成 8922/1555、来源组零重叠的开发划分；官方 3750
  张 test 无公开标签，最终必须导出 COCO JSON 并提交服务器。
- 正式训练锁已于 2026-08-20 23:29 UTC 固化为
  `formal_experiment_lock.json`，状态为 `locked_authorized`；锁定哈希、
  远端环境、权重、数据协议和划分计数均通过预检。计时校准和三轮排错
  预检不得混入论文结果。
- 12 个 seed-0 baseline 队列已于 2026-08-20 23:34 UTC 启动，正式输出
  位于 `/root/autodl-tmp/formal_baselines`，状态位于
  `/root/autodl-tmp/formal_state/baseline_queue_state.json`。

## 推荐执行顺序

1. 审核并启用统一训练锁。（已完成）
2. 跑 12 个 seed-0 modern baselines。（运行中）
3. SeaDronesSee 按 E1、E2、E3、E4 逐关推进；无增益则停止下一关。
4. 方法冻结后迁移到 YOLO26，再跑三数据集 final。
5. 最后才跑 baseline/final 的 seed 1/2 与一次 MDC-YOLO11。
6. 第一周末根据 E2/E3 结果决定续租第二周或增加第二张 4090。
