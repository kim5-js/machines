# FloW 最新来源划分：CBAM/Focal 五组、三 seed

用户于 2026-09-20 明确确认五组 × seeds 0/1/2，与同一新划分的 E0/E2/E3 配对。

数据固定为 FloW_IMG_YOLO_2000_source_disjoint：1124 张 train、335 张 val、541 张 test；类别 bottle。沿用来源分组协议与已修正标注，所有原始图片/标签和成员清单按 SHA-256 核对。已有元数据支持来源分离；不声称未知原始视频 ID 的绝对分离。

五组按 CBAM、Focal-EIoU、CBAM+Focal-EIoU、组合+E2、组合+E3 顺序，每组 seeds 0、1、2。E0/E2/E3 复用 9 月 20 日完成的同划分、同 seed 结果，不重训。四个 AdamW detector baseline 不并入此 SGD 配对分析。

每次独立加载相同 YOLOv8s 预训练权重。全部训练参数复制已完成的新划分 E 系列：SGD lr0=0.006、imgsz=960、batch=8、nbs=8、150 epochs、native patience=100、AMP、相同普通增强。与早期 SeaDronesSee CBAM 的 AdamW 配方不同，本实验只复用其模块实现。保留原生早停，不为得到显著收益改变参数。

CBAM 仅插入 backbone P3/P4 的 C2f 第 4、6 层，reduction=16，kernel=7；Detect 输入保持 15/18/21。Focal-EIoU alpha=0.25、gamma=2、eps=1e-7，只替换 bbox CIoU，保留 DFL。E2 为历史随机 MDG 的 clean+degraded 双视图；E3 仅对退化视图、增强后面积小于 1024 像素平方的已分配正样本施加 1.5 权重，保持背景和归一化分母。无一致性损失、无自适应调参、无课程策略。

五组各做一次 128 train / 8 val / 2 epochs 的真实数据短训练，全通过再启动 15 组正式训练。检查每个指定模块、梯度/实际参数更新、预训练映射、数据成员和标签、实际参数、AMP、双视图目标复制与 E3 加权。新 CBAM attention 随机初始化；其余所有初始张量必须与同 seed E0 记录匹配，相同 CBAM 架构组初始张量必须一致。

best.pt 只按 335 张 val 的 native fitness 选择。每组完成后以冻结参数分别对 val/test 运行官方 pycocotools COCOeval 2.0.10（实际 batch=1 FP32、imgsz=960、conf=0.001、NMS IoU=0.7、预测 max_det=300、评价 maxDets=1/10/100），并单独保存历史项目 COCO-style 复核。native、官方 COCO、历史 COCO-style 分表；缺失值为 null，不写为零；不按 test 改参数或挑 seed。

报告逐 seed、均值、样本标准差与同 seed 配对差值；检验 CBAM/Focal 单独与组合效应、组合上加入 E2/E3 的增量，以及相对已完成 E2/E3 的差异。三 seed 证据精度有限。保留每组 best/last、CSV、参数、runtime/completion、预测与评价原件及哈希，最终回收到本地。

原 27 组旧划分队列必须训练及评价完成后才允许新队列 dispatch。唯一锁定 supervisor；不重复启动，不覆盖、不重跑完成组；失败先保存证据。数据和预训练权重均已存在远端，本次仅上传执行代码与冻结协议/来源清单。禁止 Git commit/push。
