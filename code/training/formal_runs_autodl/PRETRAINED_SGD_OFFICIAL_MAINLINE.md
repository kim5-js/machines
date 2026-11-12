# Pretrained-SGD 正式主线指定

自 2026-08-31 起，论文实验主线统一为：锁定 `yolov8s.pt` 预训练初始化、SGD、本机历史训练配方、五类无泄漏 `source_disjoint_v1` 数据划分。E0、E1、E2 和 E3 均属于这条主线；small-object weight `alpha` 是 E3 内部的超参数消融，不是新增模块。

## 与本机历史 SGD 的关系

本轮实际训练的 SGD、`lr0=0.006`、`lrf=0.01`、`momentum=0.937`、`weight_decay=6.25e-05`、warmup、box/cls/DFL 系数，以及 HSV、translate、scale、flip、mosaic、mixup、erasing 等增强参数，与本机历史 SeaDronesSee SGD run 一致。

明确保留五项差异：

1. `pretrained=true`，从锁定 `yolov8s.pt` 初始化；历史本机 run 为 scratch。
2. `batch=8`；历史本机 run 为 4。
3. `seed=0`；历史本机 run 为 42。
4. 数据改为经审计的五类 `source_disjoint_v1`（train 8922 / val 1555），不复用历史存在 source leakage 且把 ignore 当第六类的划分。
5. 运行环境改为锁定 AutoDL RTX 4090 环境。

因此，本轮是“沿用本机 SGD 训练配方的新正式主线”，但与历史本机结果的差异不能写成严格的“只改变预训练”对照。

## 主线方法顺序

- E0：单视图 pretrained YOLOv8s 基线。
- E1：E0 + train-only maritime degradation。
- E2 / alpha=1.0：clean + degraded 双视图，共享 YOLO，两路检测损失，无 consistency loss。
- E3：E2 + 仅 degraded-small assigned-positive detection loss 加权；普通目标权重 1.0。

E3 的 alpha 扫描为 1.0、1.2、1.35、1.5。现有 E2 复用为 1.0，现有 E3 复用为 1.5；1.2 已得到连续有限训练历史，1.35 尚未开始，最终 alpha 选择应等 1.35 后再定。

## 当前证据状态

| 实验 | Best epoch | Precision | Recall | mAP50 | mAP50-95 | 证据状态 |
|---|---:|---:|---:|---:|---:|---|
| E0 | 25 | 0.72415 | 0.70339 | 0.71704 | 0.47119 | 正式 early-stopping completion evidence 已通过 |
| E1 | 28 | 0.73490 | 0.71890 | 0.71821 | 0.46858 | 连续有限历史达到 patience 边界；完成标记被后处理回调阻断 |
| E2 / alpha=1.0 | 30 | 0.73203 | 0.71140 | 0.72201 | 0.47067 | 连续有限历史达到 patience 边界；完成标记被后处理回调阻断 |
| E3 / alpha=1.2 | 22 | 0.72020 | 0.71271 | 0.71726 | 0.47268 | 连续有限历史达到 epoch 122；监督器完成记账失败 |
| E3 / alpha=1.35 | - | - | - | - | - | 尚未开始 |
| E3 / alpha=1.5 | 16 | 0.72839 | 0.70732 | 0.71847 | 0.47145 | 连续有限历史达到 patience 边界；完成标记被后处理回调阻断 |

alpha=1.2 目前给出最高的单次 `mAP50-95`，但 alpha=1.35 尚未完成且目前只有 seed 0，因此只能作为主线内的机制消融证据，不能宣称稳定最优。

## 不进入主线的内容

CBAM、Focal-EIoU、consistency loss、severity weighting、curriculum、E4、scratch 队列和旧 AdamW 结果均不进入这条正式主线。没有进行跨组 checkpoint 续训，也没有因本次指定创建 Git commit 或 push。
