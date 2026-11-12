# 论文复核材料（machines-4634859，修回工作版本 20261109）

这个仓库放的是论文实验部分的全部材料：训练代码、评价代码、每次训练的记录、每次训练的权重、每项评价的输出，以及补充材料里的数据表。不含稿件、审稿意见和数据集图像。

权重、预测文件和压缩包存在 Git LFS 里。克隆前先装好 Git LFS（`git lfs install`），否则拿到的只是指针文件；全部下载约 6.4 GB。只想看代码和记录，可以用 `GIT_LFS_SKIP_SMUDGE=1 git clone ...` 跳过大文件。

## 目录

| 目录 | 内容 |
|---|---|
| `records/` | 123 次训练各自的参数和逐轮记录，533 项评价的评分记录，最初 78 项评价的封存汇总表；说明见 `records/先读我.md` |
| `checkpoints/` | 每次训练的 `best.pt` 和 `last.pt`，文件夹名与 `records/` 一一对应；`检查点索引.csv` 列出每个文件的 SHA-256 和用到它的评价 |
| `evaluation_outputs/` | 每项评价的完整输出：逐图预测（`predictions.json.gz`）、COCOeval 中间数组、评分日志和 `evaluation.json` |
| `code/training/` | 各实验块在服务器上实际运行的训练代码，按实验块分文件夹 |
| `code/evaluation/` | 统一评价脚本（官方 COCOeval）和条件评价脚本 |
| `code/analysis/` | 由评价记录生成论文的统计量、图和补充材料数据表的脚本 |
| `supplementary/` | 补充材料 S1–S6 的数据表和三个数据集的划分清单 |
| `environment/` | 服务器环境的软件版本、依赖清单和所用的 ultralytics 安装包 |

## 实验块、记录、权重和代码的对应

`records/` 和 `checkpoints/` 用同一套文件夹名。

| 文件夹 | 训练次数 | 内容 | 训练代码 |
|---|---|---|---|
| `01_检测器参照` | 12 | 四种检测器在三个数据集上各一次（种子 0） | SDS、WSODD：`formal_runs_autodl`；FloW：`flow_source_retrain_20260920` |
| `02_训练策略_SDS_WSODD` | 18 | SDS、WSODD 的标准训练和固定单视图增强 | `formal_runs_autodl` |
| `03_训练策略_FloW` | 24 | FloW 的标准训练、双视图、带小目标加权的双视图、干净重复视图对照 | `flow_source_retrain_20260920`、`flow_extension_20261109` |
| `04_匹配对照_WSODD从头训练` | 42 | 七种从头训练配置，各六个种子 | `refsoft_reproduction_20260910`、`refsoft_extension_20261109` |
| `05_匹配对照_WSODD预训练副本` | 12 | 预训练微调下追加干净副本或 GlareSoft 副本 | `pretrained_copies_20261109` |
| `06_FloW预训练单组件` | 15 | FloW 预训练微调下的五种配置，各三个种子 | `flow_source_cbam_20260920` |

记录里的名称与论文里的名称的对照，见 `records/先读我.md`。`code/training/queue_20261109` 是修回阶段追加训练的排队和启动脚本。

## 权重是怎么核对的

权重不是按文件夹名挑的，而是从评价记录出发：论文用到的每一项评价都记着所用检查点的 SHA-256，共 114 个不同的检查点（108 次训练的 `best.pt`，加上 SDS 六次训练的 `last.pt`）。这 114 个文件逐个找到、复制后重新计算哈希，全部与记录一致。`06_FloW预训练单组件` 的 15 次训练，评价记录里没有检查点哈希，文件取自该实验块回收的证据，哈希与它的回收清单一致。

其余的 `last.pt` 没有被论文评价，只是随 `best.pt` 一起放入，共 117 个。


评分用的是 pycocotools 的官方 COCOeval。评价记录里的 `evaluator_code_sha256` 是评分模块 `cocoeval.py` 的 SHA-256，78 项封存评价和 533 项后续评价记的是同一个值，与 `code/training/flow_source_retrain_20260920/vendor/pycocotools/cocoeval.py` 一致。调用它的评价脚本在 `code/evaluation/`。

## 没有放进来的东西

- 数据集图像和标注。三个数据集都是公开数据集，划分清单在 `supplementary/` 里。`evaluation_outputs/前78项/unified_evaluation_evidence.zip` 是当时封存的评价证据，里面带有三个评价划分的 COCO 格式标注文件（不含图像）。
- 训练过程中保存的示例图、曲线图和逐次训练的控制台日志。
- 稿件、审稿意见和回复信。
- `code/training/formal_runs_autodl` 里的三个文件：一份写有训练服务器登录地址的说明，两份内部的启动授权记录。

## 说明

- 代码和记录是原样复制的，里面有服务器路径、日期和内部编号；脚本里的路径指向当时的服务器或本机目录，换机器运行需要改路径。
- 带注意力模块的检查点在加载时需要能导入 `code/training/formal_runs_autodl` 里的模型定义。
- 论文里的数值来自 `evaluation_outputs/`（测试集或验证集上用官方 COCOeval 重新评分），不是训练记录 `results.csv` 里的验证指标。
