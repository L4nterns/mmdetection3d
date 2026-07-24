# MMDetection3D Fork Notes

这个 fork 保留 MMDetection3D 原仓库主体，只补充面向 labelCloud 3D 框标注数据的自动转换、训练、推理、可视化和裁切入口。

## 行为边界

自 fork 以来的改动只服务于 labelCloud 数据接入和纯点云 3D box detection，不修改 MMDetection3D 原仓库的通用训练默认值，也不把 labelCloud 数据集注册进核心 `mmdet3d/datasets` 包。

默认行为：

- 默认模型见 `.env.example`，当前为 MMDetection3D 原生 `tr3d`
- 默认从零训练，`PRETRAINED_MODEL=` 为空
- HTTP/CLI 推理默认读取 `.env` 中显式指定的 `INFER_CFG_FILE` 和 `INFER_CKPT`
- 默认训练配置由转换脚本写入 `data/labelcloud/cfgs/labelcloud_<model>.py`
- 默认转换 labelCloud `centroid_abs` 标签
- 默认点云保存为 `N x 6` 的 `.bin`：`x y z r g b`
- 默认训练配置使用 `POINT_FEATURES=xyzrgb`，即 `x y z r g b` 六维；四通道权重实验可切到 `POINT_FEATURES=xyzi`
- 默认增强模式是 `AUG_MODE=full`；`pv_rcnn` / `pointpillars` 启用 ObjectSample、随机翻转、随机旋转和尺度扰动，`tr3d` / `fcaf3d` 启用场景级翻转、旋转和平移/尺度扰动但不启用 ObjectSample
- 默认 `AUG_MODE=full` 会为 `pv_rcnn` / `pointpillars` 创建 `labelcloud_gt_database/` 和 `labelcloud_dbinfos_train.pkl`；`tr3d` / `fcaf3d` 不使用 GT database
- 默认 `POINT_CLOUD_RANGE`、anchor size 和 anchor bottom 只从 train split 统计，避免 val 信息进入训练配置
- 默认推理输出 MMDetection3D prediction package：`summary.json` 和 `<scene>/predictions.json`；传 `--crop` 或 HTTP `crop=true` 时额外输出 `crops/<scene>/`
- 默认训练重复次数和验证间隔见 `.env.example`，通过 `TRAIN_REPEAT_TIMES` 和 `VAL_INTERVAL` 控制

测试/实验开关：

- `SAMPLE_POINTS` 默认值见 `.env.example`；设置为正整数时在生成配置里插入 `PointSample`
- `SPARSE_VOXEL_SIZE` 默认值见 `.env.example`，只用于 `tr3d` / `fcaf3d` sparse RGB 配置；稠密场景显存不稳时可显式调大
- `MIN_BOX_DIM` / `MIN_POINTS_IN_GT` 默认值见 `.env.example`，转换阶段过滤近零尺寸框和框内无点的退化 GT
- label 文件中的类别名必须存在于 `_classes.json`；转换会一次性汇总所有未知类别并报错
- 过滤后如果没有有效 GT，或 train split 没有有效 GT，转换会直接失败并输出过滤统计
- `TRAIN_LR` / `TRAIN_LR_PEAK_RATIO` 默认值见 `.env.example`，生成配置默认使用保守微调学习率且不自动升高到 10 倍 LR
- `POINT_FEATURES=xyzrgb|xyzi`，默认 `xyzrgb`
- `PYTORCH_CUDA_ALLOC_CONF` 默认空；需要降低 CUDA allocator 碎片化风险时可设为 `max_split_size_mb:128`
- `AUG_MODE=full|safe|none`，默认 `full`
- `GT_DATABASE=auto|on|off`，默认 `auto`
- `MODEL=pv_rcnn|pointpillars|tr3d|fcaf3d`，默认值见 `.env.example`
- `PRETRAINED_MODEL` 只在显式填写时启用，用于外部权重初始化
- `RESUME_CKPT` 用于训练断点恢复，`INFER_CKPT` 用于推理加载
- `--skip-convert` / `--skip-train` 用于分阶段转换、检查配置和复用已准备数据
- `--gt-database auto|on|off` 可在转换时覆盖 DB 生成策略
- `--skip-gt-database` 会同时从生成配置里移除 `ObjectSample`，避免训练引用不存在的 `labelcloud_dbinfos_train.pkl`
- `--rebuild-gt-database` 会强制重建 DB；默认 DB 支持按源文件 mtime 续跑并跳过未过期目标

## 使用场景

当前目标是港口堆场货物点云拆分：

- 输入：labelCloud 导出的 `_classes.json`、`pointClouds/`、`labels/`
- 标注：3D box，类别从 `_classes.json` 自动读取
- 点云特征：直接读取 `.pcd` 的 `x y z r g b`；无色点云的 RGB 填 0
- 输出：MMDetection3D 3D 检测结果写入 `predictions.json`，后续按需可视化或裁切

这不是点级语义分割或实例分割流程；当前 fork 只做 3D box detection。

## 模型选择

OpenPCDet fork 中自动化入口支持：

```text
pv_rcnn
pv_rcnn_plusplus
dsvt_pillar
dsvt_voxel
```

迁移到 MMDetection3D 后，当前自动化入口支持：

```text
pv_rcnn
pointpillars
tr3d
fcaf3d
```

默认选择 `tr3d`，原因是：

- 港口堆场点云是稠密静态 `XYZRGB` 场景，TR3D 的 sparse-conv 路线更贴近 ScanNet-style 室内/场景点云假设
- 默认不依赖 labelCloud GT database/ObjectSample，训练准备流程更简单
- 目标是稳定得到可裁切的 3D box，不是复现 OpenPCDet 的全部模型 zoo
- `pv_rcnn` / `pointpillars` 仍作为 LiDAR-style 或调试/基线模型保留

`tr3d` 和 `fcaf3d` 是为稠密 `XYZRGB` 场景点云新增的 sparse-conv 路线，面向 ScanNet-style 预训练权重微调；它们使用 `DEPTH` box/coord 配置、`MinkowskiEngine`，并且不使用 labelCloud GT database/ObjectSample。

没有在 MMDetection3D 中硬迁移 `pv_rcnn_plusplus`、`dsvt_pillar`、`dsvt_voxel`。这些模型在两个项目里的实现、配置结构和依赖并不一一对应，直接生成同名配置会变成不可验证的承诺。

## 保留的改动

1. labelCloud 数据集插件

- `projects/labelcloud/labelcloud_dataset.py`
  - 注册 `LabelCloudDataset`
  - 支持 `_classes.json` 中的任意类别名
  - 读取转换脚本生成的 MMDetection3D V2 pkl 标注
  - labelCloud box 按几何中心写入，数据集解析时转换为 MMDetection3D LiDAR box 使用的底中心语义
- `projects/labelcloud/metrics.py`
  - 注册 `LabelCloudMetric`
  - 验证阶段直接评估转换后的 LiDAR 3D box，报告 `labelcloud/mAP@0.25`、`labelcloud/mAP@0.5`、precision、recall、F1 和 per-class AP
  - 不把自定义 labelCloud 数据伪装成 KITTI 格式评测；指标语义只对应本 fork 的 class-aware 3D IoU/AP
- `projects/labelcloud/utils.py`
  - 共享 labelCloud 类别读取、PCD 读取、box 解析、点云裁切和原子写文件工具

2. labelCloud 转换脚本

- `tools/convert_labelcloud_to_custom.py`
  - 读取 `_classes.json` 获取类别，不在脚本里写死类别
  - 读取 labelCloud `centroid_abs` 格式导出的 JSON 标签
  - labelCloud `centroid_abs` 语义按 `center + length/width/height + rot_z` 处理
  - 将 labelCloud 的 `rot_z` 从度转换为 yaw 弧度 `[-pi, pi]`，不做符号反转
  - 对 x/y 轴旋转做显式报错，因为当前训练配置只支持 yaw-only 3D box
  - 转换点云前先校验 scene id 唯一性，避免同名 PCD/重复 filename 静默覆盖转换产物
  - 读取 `.pcd` 并保存为 `N x 6` 的 `.bin`：`x y z r g b`
  - 按固定随机种子划分训练集 `train` 和验证集 `val`
  - 生成 `labelcloud_infos_train.pkl`、`labelcloud_infos_val.pkl`、`labelcloud_infos_trainval.pkl`
  - `pv_rcnn` / `pointpillars` 默认只有 `AUG_MODE=full` 时生成 `labelcloud_gt_database/` 和 `labelcloud_dbinfos_train.pkl`
  - `--gt-database on|off` 可强制开启/关闭 DB
  - GT database 支持按源文件 mtime 续跑、点分块扫描、过期目标重写和 `--gt-database-max-points` 单目标限点
  - 生成模型配置：`data/labelcloud/cfgs/labelcloud_<model>.py`
  - 支持 `--model pv_rcnn|pointpillars|tr3d|fcaf3d`
  - 支持 `--point-features xyzrgb|xyzi`；推理脚本会读取生成配置里的 `use_dim` 保持一致
  - `pointpillars` 配置将 `voxel_size[2]` 设为完整 z range，确保 BEV pillar 只有 1 个 z cell
  - `tr3d` / `fcaf3d` 配置使用 `coord_type='DEPTH'`、`box_type_3d='Depth'`、`use_color=True` 和 `XYZRGB` 输入
  - `tr3d` / `fcaf3d` 不创建或引用 GT database；即使 `AUG_MODE=full`，也不会插入 `ObjectSample`
  - `tr3d` / `fcaf3d` 不在代码里硬编码点数采样；只有 `SAMPLE_POINTS` / `--sample-points` 为正整数时才插入采样步骤
  - 默认增量转换：已存在且未过期的点云 `.bin` 会跳过；`--overwrite` 会清理并重建生成物

3. 一键训练入口

- `tools/train_labelcloud_pipeline.py`
  - 串联转换和训练
  - 默认训练配置跟随 `--out-dir` 自动派生
  - 默认 `--num-gpus auto`，使用当前环境可见的所有 GPU
  - 多卡时通过 `torch.distributed.run` 启动 MMDetection3D 分布式训练
  - 默认 `--batch-size auto`，按每张 GPU 2 个样本计算总 batch；需要更小 batch 时在 `.env` 里设置 `BATCH_SIZE`
  - 支持 `--train-repeat-times` / `TRAIN_REPEAT_TIMES`
  - 支持 `--val-interval` / `VAL_INTERVAL`
  - 支持 `--model pv_rcnn|pointpillars|tr3d|fcaf3d`
  - 支持 `--sample-points` / `SAMPLE_POINTS`
  - 支持 `--point-features` / `POINT_FEATURES`
  - 支持 `--aug-mode` / `AUG_MODE`
  - 支持 `--gt-database` / `GT_DATABASE`、`--skip-gt-database`、`--rebuild-gt-database` 和 DB 分块/限点参数
  - `--workers 0` 会同步关闭 dataloader `persistent_workers`
  - 支持 `--pretrained-model` / `PRETRAINED_MODEL`
  - 支持 `--resume-ckpt` / `RESUME_CKPT`
  - 支持 `--skip-convert`、`--skip-train`、`--reset-output`、`--dry-run`

4. 推理、裁切和可视化

- `tools/infer_labelcloud.py`
  - 读取 `.pcd`，转为 numpy 点云后调用 `mmdet3d.apis.inference_detector`
  - 通过 MMEngine 配置对象读取 `use_dim`，保证推理点特征与训练配置一致
  - 支持目录或单个 `.pcd`
  - `--ckpt` 必须是明确 checkpoint 路径，不从训练 `work_dirs` 自动查找 checkpoint
  - 默认输出每个场景的 `predictions.json` 和全局 `summary.json`
  - 传 `--crop` 时输出 `crops/<scene>/`，并在 `predictions.json` 每个对象内写入 `crop_path`
- `tools/http_infer_labelcloud.py`
  - 提供 FastAPI HTTP 推理服务，接口为 `POST /v1/infer`
  - 复用 `tools/infer_labelcloud.py` 的点特征维度解析和预测 JSON 生成逻辑
  - 模型配置、checkpoint 和 device 只从服务环境变量读取，不允许请求方指定本地路径
  - `INFER_CKPT` 必须是明确 checkpoint 路径，常驻 HTTP 服务不使用自动查找 checkpoint
  - 模型按 `model/cfg/ckpt/device` 懒加载并缓存在服务进程内，避免每个请求重复初始化
  - 请求字段 `scene_pcd` 接收上传的场景 `.pcd`，`score_thresh` 控制本次响应分数阈值，`crop=true` 时生成对象裁切 PCD
  - 同步返回 zip；默认包含 `summary.json` 和 `<scene>/predictions.json`，`crop=true` 时额外包含 `crops/<scene>/`
  - 请求处理使用临时目录保存中间文件，响应结束后自动删除，不持久化推理文件
  - 服务进程内不做推理全局串行锁；需要多 GPU 并发时建议按 GPU 启多个服务进程，由上游做负载均衡
- `tools/crop_labelcloud_predictions.py`
  - 读取 `predictions.json`，从原始 `.pcd` 按旋转 3D box 裁切目标点云
  - 输出每个目标的 `.pcd` 和 `crop_summary.json`
- `tools/visualize_labelcloud_result.py`
  - 读取 `predictions.json` 和原始 `.pcd`
  - 使用 Open3D 显示点云和预测 3D box

## 数据目录

准备 labelCloud 数据目录，目录内包含：

```text
labelCloud/
  _classes.json
  pointClouds/
  labels/
```

转换后的默认目录：

```text
data/labelcloud/
  points/
  ImageSets/
  labelcloud_infos_train.pkl
  labelcloud_infos_val.pkl
  labelcloud_infos_trainval.pkl
  labelcloud_gt_database/
  labelcloud_dbinfos_train.pkl
  labelcloud_gt_database_manifest.json
  conversion_summary.json
  cfgs/
    labelcloud_<model>.py
```

转换脚本一次生成当前 `--model` 对应的配置。需要同时保留多个模型配置时，分别用不同 `--model` 执行转换；共享的 `points/`、`ImageSets/` 和 infos 会按增量逻辑复用。

`pv_rcnn` / `pointpillars` 默认 `AUG_MODE=full` 时会生成 `labelcloud_gt_database/` 和 `labelcloud_dbinfos_train.pkl`；如果切到 `AUG_MODE=safe|none`，DB 目录只会在 CLI 显式传 `--gt-database on` 时出现。`tr3d` / `fcaf3d` 不支持 `--gt-database on`。

## Docker/Compose 用法

这个 fork 增加了 labelCloud 专用 Compose 入口，复用 mmdetection3d 源码构建镜像，不把宿主机整个仓库挂载到容器内源码目录。默认数据挂载：

```text
./labelCloud  -> /workspace/labelCloud
./infer       -> /workspace/infer
./checkpoints -> /workspace/checkpoints
./data        -> /workspace/mmdetection3d/data
./output      -> /workspace/mmdetection3d/output
```

准备运行配置：

```bash
cp .env.example .env
```

`.env` 是唯一的 Compose 环境配置。训练/转换使用 `CFG_FILE`；HTTP/CLI 推理使用 `INFER_CFG_FILE` 和 `INFER_CKPT`，因此可以在同一个 `.env` 中同时保留训练参数和推理配置。

构建镜像：

```bash
docker compose build
```

常用服务：

```text
infer-http  默认常驻 HTTP 推理服务
pipeline    jobs profile，convert + train
convert     jobs profile，只转换 labelCloud 数据并生成当前模型配置
reconvert   jobs profile，清理并重建转换产物
prepare     jobs profile，只转换并按当前模型策略准备数据，不训练
reprepare   jobs profile，重建转换产物/DB，不训练
train       jobs profile，基于已有转换结果训练
retrain     jobs profile，清理当前 work-dir 后重新训练
infer       jobs profile，对 infer/ 里的 .pcd 推理
shell       tools profile，进入容器
```

默认 `docker compose up -d` 只启动 `infer-http`。训练、转换和 CLI 推理是一次性任务，使用 `docker compose --profile jobs run --rm <service>` 显式执行；调试 shell 使用 `tools` profile。

训练完成并确认模型可用后，把用于推理的 cfg 和 checkpoint 复制到固定推理目录，再启动或重启推理服务：

```bash
mkdir -p checkpoints/deploy/tr3d_current
cp data/labelcloud/cfgs/labelcloud_tr3d.py checkpoints/deploy/tr3d_current/config.py
cp data/labelcloud/work_dirs/labelcloud_tr3d/<best_or_epoch_checkpoint>.pth checkpoints/deploy/tr3d_current/model.pth
```

HTTP 推理配置示例：

```text
INFER_CFG_FILE=/workspace/checkpoints/deploy/tr3d_current/config.py
INFER_CKPT=/workspace/checkpoints/deploy/tr3d_current/model.pth
```

常驻 HTTP 推理服务使用固定推理目录里的 `config.py` 和 `model.pth`；`reconvert`、`retrain` 或新一轮 `pipeline` 可能重建训练目录里的 `data/labelcloud/cfgs` 和 `work_dirs`，不要把它们作为服务配置源。

启动 HTTP 推理：

```bash
docker compose up -d
```

一键转换并训练：

```bash
docker compose --profile jobs run --rm pipeline
```

只转换：

```bash
docker compose --profile jobs run --rm convert
```

只训练：

```bash
docker compose --profile jobs run --rm train
```

推理：

```bash
docker compose --profile jobs run --rm infer
```

进入调试 shell：

```bash
docker compose --profile tools run --rm shell
```

## HTTP 推理接口

启动 HTTP 推理服务：

```bash
docker compose up -d
```

HTTP 服务环境变量见 `.env.example`。脚本直接运行时也会自动读取当前工作目录或仓库根目录的 `.env`；已存在的系统环境变量不会被 `.env` 覆盖。

HTTP endpoint：

```text
POST /v1/infer
GET  /health
```

HTTP 请求表单字段：

```text
scene_pcd     必填，上传单个 .pcd 场景点云
score_thresh  可选，0 到 1 之间的分数阈值；默认读取 SCORE_THRESH
crop          可选，true/false；默认读取 HTTP_CROP，HTTP_CROP 默认 false
request_id    可选，用于响应头、临时目录和返回 zip 文件名的安全前缀，不会持久化
```

设置 `HTTP_AUTH_TOKEN` 后，`POST /v1/infer` 请求必须带 `Authorization: Bearer <token>`；不设置时不启用鉴权。

HTTP 请求不能覆盖模型、checkpoint、GPU 或本地路径相关配置；这些配置全部由服务启动环境控制。需要调整时，改 `.env` 或进程环境后重启服务。

HTTP 推理示例：

```bash
curl -X POST http://127.0.0.1:8011/v1/infer \
  -H "Authorization: Bearer $HTTP_AUTH_TOKEN" \
  -F "scene_pcd=@infer/example.pcd" \
  -o result.zip
```

需要对象裁切 PCD 时：

```bash
curl -X POST http://127.0.0.1:8011/v1/infer \
  -H "Authorization: Bearer $HTTP_AUTH_TOKEN" \
  -F "scene_pcd=@infer/example.pcd" \
  -F "crop=true" \
  -o result_with_crops.zip
```

HTTP 成功返回 `application/zip`，响应头包含 `X-Request-ID`。zip 文件名为 `<request_id>_mmdetection3d.zip`；未传 `request_id` 时使用服务生成的随机 ID。zip 内容：

```text
summary.json
<scene>/predictions.json
crops/<scene>/*.pcd  # 仅 crop=true 时存在
```

HTTP 错误返回 JSON：

```json
{
  "ok": false,
  "error": {
    "code": "bad_request",
    "message": "scene_pcd 只支持 .pcd 文件。"
  },
  "request_id": "example"
}
```

多 GPU HTTP 部署建议按“一张物理 GPU 一个服务进程”拆分，进程内统一使用 `DEVICE=cuda:0`，通过 `CUDA_VISIBLE_DEVICES` 选择物理卡。例如：

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 HTTP_PORT=8011 python tools/http_infer_labelcloud.py
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 HTTP_PORT=8012 python tools/http_infer_labelcloud.py
```

不要用一个进程里暴露多个 GPU 再让请求传 `device`；mmd3d HTTP 请求不接受本地路径或 device 参数。

Docker 推理默认读取宿主机 `INFER_DIR=./infer` 挂载到容器内的 `/workspace/infer`。如需在容器内指定另一个输入路径，可以覆盖 `INPUT_DIR`：

```bash
INPUT_DIR=/workspace/infer PRED_OUT_DIR=output/predictions/custom_run docker compose --profile jobs run --rm infer
```

默认 `docker-compose.yml` 面向 release/reproducible 运行，不挂载宿主机源码目录，容器执行镜像构建时复制并安装的代码。

默认镜像文件是 `docker/Dockerfile.labelcloud-cu116`，默认基础镜像是 `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel`。TR3D/FCAF3D 依赖的 `MinkowskiEngine` 会在镜像构建阶段安装；需要换 CUDA/PyTorch、MinkowskiEngine 版本或公司内网源时，在 `.env` 或构建时环境变量里覆盖 `BASE_IMAGE`、`MINKOWSKI_ENGINE_VERSION`、`APT_MIRROR`、`PIP_INDEX_URL` 和 `PIP_TRUSTED_HOST`。

## 命令行用法

mmdetection3d 侧的入口也可以作为普通 Python 脚本直接运行，便于复用 MMDetection3D 原仓库的安装、训练和分布式启动方式。

只转换数据和生成配置：

```bash
python3 tools/convert_labelcloud_to_custom.py \
  --labelcloud-root ./labelCloud \
  --out-dir data/labelcloud \
  --model pv_rcnn
```

也可以显式指定三类输入目录：

```bash
python3 tools/convert_labelcloud_to_custom.py \
  --class-file ./labelCloud/_classes.json \
  --pointcloud-dir ./labelCloud/pointClouds \
  --label-dir ./labelCloud/labels \
  --out-dir data/labelcloud \
  --model pv_rcnn
```

转换并训练：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --out-dir data/labelcloud \
  --model pv_rcnn
```

使用轻量模型调试：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --out-dir data/labelcloud \
  --model pointpillars
```

分阶段执行：

```bash
# 只转换，不训练
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --skip-train

# 已有转换结果，只重新训练
python3 tools/train_labelcloud_pipeline.py \
  --skip-convert \
  --model-cfg data/labelcloud/cfgs/labelcloud_pv_rcnn.py
```

重建生成物：

```bash
# 清理并重建 points、ImageSets、infos、GT database 和 summary
python3 tools/convert_labelcloud_to_custom.py \
  --labelcloud-root ./labelCloud \
  --overwrite

# 一键入口中透传 overwrite
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --overwrite
```

数据划分：

```text
--train-ratio 0.8
--val-ratio 0.2
--seed 42
```

转换脚本生成配置时可用的几何和采样参数：

```text
--voxel-xy 0.1
--range-margin 5.0
--min-points-filter 5
--sample-group-size 15
```

这些参数直接传给 `tools/convert_labelcloud_to_custom.py`。`voxel-xy` 和 `range-margin` 用于从训练集标签范围派生 `point_cloud_range` 和 voxel size；`min-points-filter`、`sample-group-size` 用于 `pv_rcnn` / `pointpillars` 生成 `ObjectSample` 的 DB sampler 配置。

## 训练参数

模型选择：

```text
--model pv_rcnn       MMDetection3D 原生 PV-RCNN
--model pointpillars  轻量调试/基线模型
--model tr3d          稠密 XYZRGB 场景点云优先模型，需要 MinkowskiEngine；当前默认值见 .env.example
--model fcaf3d        稠密 XYZRGB 场景点云备选模型，需要 MinkowskiEngine
```

批量大小和 GPU：

```text
--num-gpus auto       使用当前环境可见 GPU 数
--batch-size auto     总 batch size = max(1, GPU 数) * 2
--workers 2           dataloader workers，设为 0 时同步关闭 persistent_workers
--master-port PORT    多卡 torch.distributed.run 端口
```

`--batch-size` 按 OpenPCDet fork 入口习惯表示总 batch size；多卡训练时脚本会换算成 MMDetection3D/MMEngine 的单卡 `train_dataloader.batch_size`。

训练轮数和输出目录：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --epochs 80 \
  --work-dir data/labelcloud/work_dirs/labelcloud_pv_rcnn
```

`--reset-output` 只清理当前 `--work-dir`，不会清理转换数据或其他实验输出：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --reset-output
```

查看一键入口实际会执行的转换和训练命令：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --dry-run
```

## 增强和 GT Database

增强模式：

```text
AUG_MODE=full  默认值；pv_rcnn/pointpillars 启用 ObjectSample，tr3d/fcaf3d 只做场景级增强
AUG_MODE=safe  只保留尺度扰动，适合局部扫描和小数据排错
AUG_MODE=none  不做 gt database 采样、翻转、旋转或尺度扰动
```

增强和采样参数在转换阶段写入生成的 `data/labelcloud/cfgs/labelcloud_<model>.py`。修改 `.env` 里的 `AUG_MODE`、`POINT_FEATURES`、`SAMPLE_POINTS` 或 `SPARSE_VOXEL_SIZE` 后，需要重新执行 `pipeline`、`convert` 或 `reconvert` 生成配置；只执行 `train` / `retrain` 会复用已有配置。

命令行等价写法：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --aug-mode full
```

GT database 仅适用于 `pv_rcnn` / `pointpillars`，由 `AUG_MODE=full` 默认派生。需要调试或预生成 DB 时，可以用 CLI 覆盖：

```text
--gt-database auto  pv_rcnn/pointpillars 只有 AUG_MODE=full 时生成 DB，默认值
--gt-database on    即使 safe/none 也生成 DB，便于预先准备
--gt-database off   不生成 DB
```

`tr3d` / `fcaf3d` 不使用 GT database，`--gt-database on` 会直接报错。

命令行等价写法：

```bash
python3 tools/convert_labelcloud_to_custom.py \
  --labelcloud-root ./labelCloud \
  --aug-mode full \
  --gt-database auto
```

对 `pv_rcnn` / `pointpillars`，`full` 需要 `labelcloud_dbinfos_train.pkl`。如果使用 `--skip-gt-database`，生成配置会自动移除 `ObjectSample`；即使 `AUG_MODE=full`，也不会引用不存在的 GT database。`tr3d` / `fcaf3d` 始终不引用 GT database。

GT database 默认按源文件 mtime 续跑，已存在且不早于源 PCD、label、class 文件的单目标 `.bin` 会跳过。强制重建：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --aug-mode full \
  --rebuild-gt-database
```

分块和限点参数：

```text
--gt-database-target-elements 4000000
--gt-database-min-chunk-size 50000
--gt-database-max-chunk-size 500000
--gt-database-max-points 0
```

`target-elements` 是每次点框筛选的目标计算量，近似等于 `当前场景缺失 box 数 x chunk 点数`。如果场景点数很大，可以降低它来减少单次内存峰值：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --aug-mode full \
  --gt-database-target-elements 1000000
```

## 点特征和预训练

点特征：

```text
POINT_FEATURES=xyzrgb  使用 x/y/z/r/g/b 六维，默认值
POINT_FEATURES=xyzi    使用 x/y/z/r 四维，用于四通道权重实验或消融
```

命令行等价写法：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --point-features xyzi
```

`xyzrgb` 会生成 6 通道模型配置；常见 KITTI PV-RCNN 权重通常是 4 通道输入，使用这类权重时应切到 `xyzi`，或者只加载 shape 匹配的外部权重。

外部预训练初始化：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --point-features xyzi \
  --pretrained-model checkpoints/pv_rcnn_kitti.pth
```

权重文件路径按训练进程当前工作目录解析。推荐把权重放到宿主机 `CHECKPOINTS_DIR=./checkpoints`：

```text
宿主机直接训练：
PRETRAINED_MODEL=./checkpoints/tr3d_1xb16_scannet-3d-18class.pth

Docker Compose 训练：
PRETRAINED_MODEL=/workspace/checkpoints/tr3d_1xb16_scannet-3d-18class.pth
```

只写文件名（例如 `PRETRAINED_MODEL=tr3d_1xb16_scannet-3d-18class.pth`）时，程序会在当前工作目录查找；Docker Compose 的当前工作目录是 `/workspace/mmdetection3d`，不会自动去 `/workspace/checkpoints` 找。

断点续训：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --skip-convert \
  --model-cfg data/labelcloud/cfgs/labelcloud_pv_rcnn.py \
  --resume-ckpt data/labelcloud/work_dirs/labelcloud_pv_rcnn/epoch_20.pth
```

`RESUME_CKPT` 续训规则：

```text
RESUME_CKPT=auto      仅 train 服务和 --skip-convert 训练入口生效，传给 tools/train.py --resume auto，由 MMEngine 从 WORK_DIR 自动恢复最新 checkpoint
RESUME_CKPT=          不续训，不向 tools/train.py 传 --resume
RESUME_CKPT=<path>    从指定 checkpoint 续训，恢复 epoch/optimizer/scheduler
```

完整转换+训练流程会重新准备数据和配置，因此遇到 `RESUME_CKPT=auto` 会忽略自动续训，避免新数据准备后误接旧输出。需要在完整流程后恢复指定 checkpoint 时，显式写具体 `RESUME_CKPT` 路径。

`docker compose --profile jobs run --rm train` 默认使用 `.env` 中的 `RESUME_CKPT=auto` 行为。

`docker compose --profile jobs run --rm retrain` 会先清理 `WORK_DIR`，因此该服务只在本次容器运行环境中把 `RESUME_CKPT` 覆盖为空，不会修改 `.env`。需要清空输出后从外部 checkpoint 恢复时，把 checkpoint 放到 `CHECKPOINTS_DIR` 之类的外部目录，并在执行命令时显式覆盖 `RESUME_CKPT`。

`PRETRAINED_MODEL`、`RESUME_CKPT` 和 `INFER_CKPT` 的区别：

```text
PRETRAINED_MODEL  外部权重初始化，写入 cfg.load_from，适合微调
RESUME_CKPT       训练断点恢复，传给 tools/train.py --resume，适合继续同一实验
INFER_CFG_FILE    推理加载配置，HTTP/CLI 推理使用明确配置路径
INFER_CKPT        推理加载权重，HTTP 服务使用明确 checkpoint 路径
```

## 点数采样

默认每帧点数采样上限：

```text
SAMPLE_POINTS=500000
```

这个默认对所有模型一致，包括 `tr3d` / `fcaf3d`。如需不限制每帧点数，可在 `.env` 或命令行设为 `-1`。

`tr3d` / `fcaf3d` 还支持 sparse RGB voxel size：

```text
SPARSE_VOXEL_SIZE=0.015
```

默认值见 `.env.example`。TR3D 在稠密大场景里 OOM 时，可保持 `SAMPLE_POINTS` 不变并适当调大 `SPARSE_VOXEL_SIZE`；这会减少 sparse conv 的活跃 voxel 数，通常比继续降低点数更稳定。

限制每帧点数：

```bash
SAMPLE_POINTS=200000 python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --model pv_rcnn
```

稠密 `XYZRGB` 场景使用 TR3D 时同样通过 `.env` 或 CLI 显式设置：

```bash
MODEL=tr3d SAMPLE_POINTS=500000 SPARSE_VOXEL_SIZE=0.015 BATCH_SIZE=1 python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud
```

命令行等价写法：

```bash
python3 tools/train_labelcloud_pipeline.py \
  --labelcloud-root ./labelCloud \
  --sample-points 200000
```

`sample-points` 会在生成配置中插入 `PointSample`，影响训练和测试 pipeline 看到的点数。它是显存、速度和质量之间的权衡项。

## 训练稳定性

生成配置默认启用梯度裁剪：

```text
optim_wrapper.clip_grad=dict(max_norm=10, norm_type=2)
```

`tr3d` / `fcaf3d` 和其他生成配置都会从 `.env` 读取训练学习率：

```text
TRAIN_LR=0.0003
TRAIN_LR_PEAK_RATIO=1.0
TRAIN_REPEAT_TIMES=auto
VAL_INTERVAL=5
TRAIN_AMP=false
TRAIN_AMP_DTYPE=fp16
```

`TRAIN_LR_PEAK_RATIO=1.0` 表示第一段 CosineAnnealingLR 不把学习率拉高；如显式设成 10，第一段会升到 `TRAIN_LR * 10`。自定义单类别、小数据或从 ScanNet 权重微调时，不建议默认使用 10 倍升 LR。

`TRAIN_REPEAT_TIMES=auto` 会按模型选择训练集重复次数：`tr3d/fcaf3d=5`，`pv_rcnn/pointpillars=2`。`VAL_INTERVAL=5` 表示 80 epoch 训练中每 5 个 epoch 验证一次；`VAL_RATIO=0` 时不会执行验证。

`TRAIN_AMP=true` 会把训练入口切到 MMEngine AMP。`TRAIN_AMP_DTYPE=fp16` 使用动态 loss scale；`TRAIN_AMP_DTYPE=bf16` 使用 bfloat16 autocast 且不使用 loss scale。

转换阶段会过滤明显不适合训练的 GT：

```text
MIN_BOX_DIM=0.0001
MIN_POINTS_IN_GT=1
```

`MIN_BOX_DIM` 过滤任意边长过小的 3D 框；`MIN_POINTS_IN_GT` 过滤原始点云中框内点数不足的 3D 框。过滤数量会写入 `conversion_summary.json` 的 `dropped_boxes_by_reason` 和 `dropped_boxes_by_class`。如需完整保留原始标注做排查，可把对应值设为 `0`。

## 推理

准备推理输入：

```text
infer/
  geomap_20260601_130101.pcd
  geomap_20260601_130245.pcd
```

默认推理：

```bash
python3 tools/infer_labelcloud.py \
  --model pv_rcnn \
  --cfg-file data/labelcloud/cfgs/labelcloud_pv_rcnn.py \
  --ckpt checkpoints/deploy/pv_rcnn_current/model.pth \
  --input-dir ./infer \
  --output-dir output/predictions/labelcloud_pv_rcnn
```

需要对象裁切 PCD 时传 `--crop`：

```bash
python3 tools/infer_labelcloud.py \
  --model pv_rcnn \
  --cfg-file data/labelcloud/cfgs/labelcloud_pv_rcnn.py \
  --ckpt checkpoints/deploy/pv_rcnn_current/model.pth \
  --input-dir ./infer \
  --output-dir output/predictions/labelcloud_pv_rcnn \
  --crop
```

指定 checkpoint：

```bash
python3 tools/infer_labelcloud.py \
  --cfg-file data/labelcloud/cfgs/labelcloud_pv_rcnn.py \
  --ckpt checkpoints/deploy/pv_rcnn_current/model.pth \
  --input-dir ./infer
```

指定设备：

```bash
python3 tools/infer_labelcloud.py \
  --cfg-file data/labelcloud/cfgs/labelcloud_pv_rcnn.py \
  --ckpt checkpoints/deploy/pv_rcnn_current/model.pth \
  --device cuda:0
```

启用推理 AMP：

```bash
INFER_AMP=true INFER_AMP_DTYPE=fp16 python3 tools/infer_labelcloud.py \
  --cfg-file data/labelcloud/cfgs/labelcloud_pv_rcnn.py \
  --ckpt checkpoints/deploy/pv_rcnn_current/model.pth \
  --input-dir ./infer
```

指定推理分数阈值：

```bash
python3 tools/infer_labelcloud.py \
  --cfg-file data/labelcloud/cfgs/labelcloud_pv_rcnn.py \
  --ckpt checkpoints/deploy/pv_rcnn_current/model.pth \
  --input-dir ./infer \
  --score-thresh 0.2
```

默认 `SCORE_THRESH=0.3`，低于阈值的预测框不会写入每个场景的 `predictions.json`。

把自动拆分出的 `val` 评估集复制到 `infer/` 做 `.pcd` 推理：

```bash
mkdir -p infer
find infer -maxdepth 1 -type f -name '*.pcd' -delete

while IFS= read -r id; do
  cp "labelCloud/pointClouds/${id}.pcd" "infer/${id}.pcd"
done < data/labelcloud/ImageSets/val.txt
```

推理输出；传 `--crop` 时会额外生成 `crops/`：

```text
output/predictions/labelcloud_pv_rcnn/
  summary.json
  geomap_20260601_130101/
    predictions.json
  crops/
    geomap_20260601_130101/
      000_ClassA_0.9321.pcd
```

`predictions.json` 中的 box 使用 labelCloud/后处理友好的几何中心语义：

```text
x y z dx dy dz heading
```

其中 `z` 是几何中心；训练内部使用 MMDetection3D LiDAR box 的 bottom center，推理脚本输出时会转回几何中心。

传 `--crop` 或 HTTP `crop=true` 时，`predictions.json` 的每个对象会带 `crop_path`，指向同一输出目录下的 `crops/<scene>/<object>.pcd`。后处理只需要消费这个 MMDetection3D prediction package。

## 人工修正和手动分割

MMDetection3D 不直接维护 labelCloud 人工修正流程。需要在 labelCloud 项目和 MMDetection3D prediction package 之间互转时，使用独立的 `labelcloud-m3d-converter` 项目。

## 可视化和裁切

宿主机可视化和裁切可以用独立的 `uv` 项目管理 Open3D 等 GUI/查看依赖：

```bash
uv sync --project tools/labelcloud_host --group labelcloud-host
```

可视化整帧点云并叠加预测框：

```bash
uv run --project tools/labelcloud_host --group labelcloud-host python tools/visualize_labelcloud_result.py \
  output/predictions/labelcloud_pv_rcnn/geomap_20260601_130101
```

显式指定原始整帧点云：

```bash
uv run --project tools/labelcloud_host --group labelcloud-host python tools/visualize_labelcloud_result.py \
  output/predictions/labelcloud_pv_rcnn/geomap_20260601_130101 \
  --input-file infer/geomap_20260601_130101.pcd
```

只显示分数不低于 `0.2` 的预测框：

```bash
uv run --project tools/labelcloud_host --group labelcloud-host python tools/visualize_labelcloud_result.py \
  output/predictions/labelcloud_pv_rcnn/geomap_20260601_130101 \
  --score-thresh 0.2
```

按预测框裁切目标点云：

```bash
uv run --project tools/labelcloud_host --group labelcloud-host python tools/crop_labelcloud_predictions.py \
  output/predictions/labelcloud_pv_rcnn/geomap_20260601_130101
```

指定裁切输出目录：

```bash
uv run --project tools/labelcloud_host --group labelcloud-host python tools/crop_labelcloud_predictions.py \
  output/predictions/labelcloud_pv_rcnn/geomap_20260601_130101 \
  --output-dir output/crops/labelcloud_pv_rcnn/geomap_20260601_130101
```

裁切结果：

```text
output/crops/labelcloud_pv_rcnn/geomap_20260601_130101/
  000_ClassA_0.9321.pcd
  001_ClassB_0.8123.pcd
  crop_summary.json
```

裁切是按旋转 3D box 保留框内点，不做点级实例分割或边界细化。

## 输出目录

转换后的数据：

```text
data/labelcloud/
  points/
  ImageSets/
  labelcloud_infos_train.pkl
  labelcloud_infos_val.pkl
  labelcloud_infos_trainval.pkl
  conversion_summary.json
  cfgs/
```

启用 GT database 后还会生成：

```text
data/labelcloud/
  labelcloud_gt_database/
  labelcloud_dbinfos_train.pkl
  labelcloud_gt_database_manifest.json
```

默认训练输出：

```text
data/labelcloud/work_dirs/labelcloud_pv_rcnn/
```

默认预测输出：

```text
output/predictions/labelcloud_pv_rcnn/
```

## 环境变量入口

一键训练入口支持的常用环境变量：

```text
LABELCLOUD_DIR
OUT_DIR
MODEL
CFG_FILE
TRAIN_RATIO
VAL_RATIO
SPLIT_SEED
BATCH_SIZE
WORKERS
NUM_GPUS
WORK_DIR
PRETRAINED_MODEL
RESUME_CKPT
EPOCHS
TRAIN_LR
TRAIN_LR_PEAK_RATIO
TRAIN_REPEAT_TIMES
VAL_INTERVAL
TRAIN_AMP
TRAIN_AMP_DTYPE
SAMPLE_POINTS
POINT_FEATURES
AUG_MODE
GT_DATABASE
GT_DATABASE_TARGET_ELEMENTS
GT_DATABASE_MIN_CHUNK_SIZE
GT_DATABASE_MAX_CHUNK_SIZE
GT_DATABASE_MAX_POINTS
```

其中 `MODEL` 当前支持：

```text
pv_rcnn
pointpillars
tr3d
fcaf3d
```

推理入口支持的常用环境变量：

```text
MODEL
INFER_CFG_FILE
INFER_CKPT
INPUT_DIR
PRED_OUT_DIR
SCORE_THRESH
SKIP_EXISTING
DEVICE
INFER_AMP
INFER_AMP_DTYPE
```

## 注意事项

- 这个 fork 只追加 labelCloud 纯点云 3D box detection 流程，不修改 MMDetection3D 核心数据集注册方式。
- `pv_rcnn_plusplus`、`dsvt_pillar`、`dsvt_voxel` 没有硬迁移到 MMDetection3D；自动化入口只承诺 `pv_rcnn`、`pointpillars`、`tr3d` 和 `fcaf3d`。
- 默认 Compose release 路径不挂载宿主机源码目录，避免覆盖镜像构建阶段安装好的 mmdetection3d。
- `LabelCloudMetric` 提供 class-aware 3D IoU/AP 质量评价，并默认让 checkpoint hook 按 `labelcloud/mAP@0.5` 保存最优权重。
- `data/labelcloud`、`output`、`work_dirs`、checkpoint、点云和裁切结果都是运行产物，不应作为源码改动提交。
- 最终测试集应放到 `infer/` 或通过 `--input-dir` 指定，不参与 train/val 自动划分。
- labelCloud 标签中的 x/y 轴旋转会被拒绝；当前流程只支持 yaw-only 3D box。
- 可视化和裁切脚本会从 `summary.json` 解析原始点云路径；移动过输入点云时，使用 `--input-file` 明确指定。
