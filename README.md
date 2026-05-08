# Zebrafish Pause Analysis

基于三维轨迹的斑马鱼停滞行为检测项目。
pipeline：从轨迹数据读取开始，经过预处理、特征计算、行为分段、停滞候选检测、行为分类、评估和可视化，最终输出事件级行为结果。

当前项目主要支持两类数据：

- 通用三维轨迹数据：单个标准 CSV 文件输入。
- PTZ 论文数据：原始 `PTZ_*.csv` 文件先转换为标准轨迹格式，再按鱼逐个运行检测，最后汇总评分。

## 1. 项目结构

```
configs/
  default.yaml                         # 主配置文件：路径、列名、阈值、分段、检测、分类、输出
  adaptive/
    current_dataset_thresholds.yaml    # 相对阈值模式生成的阈值覆盖文件

data/
  PTZ/                                 # PTZ 原始数据，如 PTZ_1.csv、PTZ_9.csv
  raw/
    ptz_tracks_3d_interpolated.csv     # PTZ 合并后的标准轨迹大表
    PTZ/
      ptz_1_tracks_3d_interpolated.csv # PTZ 单鱼标准轨迹文件
      ...
    zebrafish_tracks_3d_interpolated.csv

scripts/
  run_detection.py                     # 通用单文件检测入口
  run_evaluation.py                    # 通用 GT 评估入口
  run_visualization.py                 # 通用可视化入口

src/
  data_io.py                           # 配置、路径、轨迹/GT 读取和列名标准化
  preprocess.py                        # 时间戳、单位换算、低质量过滤、平滑
  features.py                          # 速度、位移、运动能量、姿态稳定性等特征
  segmentation.py                      # 变点检测和 candidate segment 构建
  detection.py                         # pause score、状态标注、停滞候选事件合并
  classification.py                    # stagnation/twist/glide 分类和冲突解决
  evaluation.py                        # 通用事件级评估
  visualization.py                     # 通用时间线和轨迹可视化
  PTZ/
    ptz_data_io.py                     # PTZ 原始 CSV 读取和单位转换
    ptz_stagnation_accuracy.py         # PTZ 专用 IM vs stagnation 评估
    ptz_stagnation_visualization.py    # PTZ 停滞事件诊断图
    run_ptz_batch_detection.py         # PTZ 单鱼批处理、汇总和自动评分入口

utils/
  convert_ptz_to_per_fish_raw.py       # PTZ 原始数据转换为标准轨迹格式
  generate_relative_thresholds.py      # 基于候选段分布生成相对阈值配置

tests/                                 # pytest 测试
outputs/                               # 运行结果输出目录
```

## 2. 标准输入格式

通用 pipeline 读取一个标准轨迹 CSV。默认路径在 `configs/default.yaml` 中配置：

```yaml
paths:
  track_file: data/raw/ptz_tracks_3d_interpolated.csv
```

标准轨迹文件至少需要以下列：

| 列名 | 含义 |
| --- | --- |
| `frame` | 帧号 |
| `id` | 鱼 ID |
| `3d_x`, `3d_y`, `3d_z` | 头部或代表点三维坐标 |
| `body_x`, `body_y`, `body_z` | 身体点三维坐标 |
| `tail_x`, `tail_y`, `tail_z` | 尾部点三维坐标 |

可选质量列包括：

```text
confidence, triangulation_error, track_quality,
interpolated, outlier_corrected
```

PTZ 数据还会保留：

```text
behavior_label
```

这列来自 PTZ 原始数据中的行为标签，用于后续把预测的 `stagnation` 与论文标注的 `IM` 进行比较。

## 3. 配置文件

主配置文件是 `configs/default.yaml`。核心配置包括：

```yaml
preprocess:
  frame_rate: 30.0
  apply_scaling: false
  smoothing_enabled: true

thresholds:
  v_th_pause: 10.0
  D_pause: 4.0
  E_th_pause: 1000.0
  S_th_pause: 0.7

detection:
  pause_enter_score: 0.65
  pause_exit_score: 0.4

adaptive:
  mode: absolute
```

当前 PTZ 标准轨迹已经在转换阶段按 `0.30 mm/pixel` 换算为毫米，所以 `apply_scaling` 应保持为 `false`。如果输入坐标仍是像素或厘米，需要根据数据单位重新设置 `apply_scaling` 和 `scale_mm_per_unit`。

`adaptive.mode` 支持两种思路：

- `absolute`：直接使用 `thresholds` 中的绝对阈值。
- `relative`：读取 `adaptive.relative_threshold_file` 中的动态阈值覆盖配置。

## 4. 通用检测流程

通用入口：

```bash
python scripts/run_detection.py --output-dir outputs/run_detection_example
```

调用关系：

```text
scripts/run_detection.py
  -> data_io.load_config()
  -> data_io.resolve_paths()
  -> data_io.load_tracking_data()
  -> data_io.standardize_tracking_columns()
  -> preprocess.preprocess_tracking_data()
  -> features.compute_all_features()
  -> segmentation.segment_behavior()
  -> detection.detect_pause_candidates()
  -> classification.classify_behavior_candidates()
```

输出结构：

```text
outputs/run_detection_example/
  final/
    behavior_events.csv
  intermediate/
    preprocess/
      preprocessed_tracks.csv
    features/
      feature_table.csv
    segmentation/
      candidate_segments.csv
      change_scores.csv
    detection/
      candidate_segments_scored.csv
      candidate_segment_states.csv
      pause_candidate_events.csv
    classification/
      stagnation_events.csv
      twist_events.csv
      glide_events.csv
      final_behavior_events_before_eval.csv
```

## 5. Pipeline 逻辑流

### 5.1 数据读取：`src/data_io.py`

负责读取配置、解析路径、读取轨迹和 GT，并把外部列名统一成内部列名。

主要函数：

```text
load_config()
resolve_paths()
load_tracking_data()
standardize_tracking_columns()
load_gt_data()
standardize_gt_columns()
```

标准化后，内部主要使用：

```text
frame, fish_id,
head_x, head_y, head_z,
body_x, body_y, body_z,
tail_x, tail_y, tail_z
```

### 5.2 预处理：`src/preprocess.py`

逻辑流：

```text
preprocess_tracking_data()
  -> add_timestamps()
  -> apply_scaling()
  -> filter_low_quality_rows()
  -> smooth_coordinates()
  -> maybe_save_preprocessed_tracks()
```

作用：

- 根据 `frame_rate` 添加 `timestamp`。
- 根据配置决定是否进行单位换算。
- 可选过滤低置信度或低质量轨迹。
- 可选对坐标做滚动平滑。

### 5.3 特征计算：`src/features.py`

逻辑流：

```text
compute_all_features()
  -> compute_velocity()
  -> compute_pose_change()
  -> compute_curvature_change()
  -> compute_window_displacement()
  -> compute_motion_energy()
  -> compute_pose_stability()
  -> compute_velocity_drop()
  -> maybe_save_feature_table()
```

主要特征：

| 特征 | 含义 |
| --- | --- |
| `v` | 头部代表点速度 |
| `p` | 头、身、尾点位移变化率 |
| `c` | 身体角度变化率 |
| `d_win` | 时间窗口内身体点位移 |
| `E_move` | 窗口内速度平方均值，表示运动能量 |
| `S_pose` | 姿态稳定性，越高表示越稳定 |
| `dv` | 速度下降幅度 |

### 5.4 行为分段：`src/segmentation.py`

逻辑流：

```text
segment_behavior()
  -> build_segmentation_features()
  -> detect_changepoints()
  -> build_candidate_segments()
  -> filter_short_segments()
  -> save_candidate_segments()
  -> save_change_scores()
```

默认用于分段的信号：

```text
v, d_win, E_move, inv_S_pose
```

其中 `inv_S_pose = 1 - S_pose`。分段模块会按鱼分组，对特征做归一化，计算相邻特征向量变化幅度，再用 `scipy.signal.find_peaks()` 找变点，最后把逐帧特征压缩为 candidate segments。

### 5.5 停滞候选检测：`src/detection.py`

逻辑流：

```text
detect_pause_candidates()
  -> score_candidate_segments()
  -> label_segment_states()
  -> merge_pause_candidates()
  -> filter_candidate_events()
  -> save_detection_outputs()
```

`score_candidate_segments()` 对每个 segment 计算四类信号：

| 信号 | 逻辑 |
| --- | --- |
| `pause_signal_v` | 速度越低越像停滞 |
| `pause_signal_d_win` | 窗口位移越低越像停滞 |
| `pause_signal_E_move` | 运动能量越低越像停滞 |
| `pause_signal_S_pose` | 姿态稳定性越高越像停滞 |

之后按配置权重合成 `pause_score`。状态标注使用进入阈值 `pause_enter_score` 和退出阈值 `pause_exit_score`，避免分数在临界值附近抖动时频繁切换。

### 5.6 行为分类：`src/classification.py`

逻辑流：

```text
classify_behavior_candidates()
  -> build_classification_inputs()
  -> classify_stagnation_events()
  -> classify_twist_segments()
  -> classify_glide_segments()
  -> merge_twist_segments_to_events()
  -> merge_glide_segments_to_events()
  -> resolve_behavior_conflicts()
  -> save_classification_outputs()
```

当前支持：

- `stagnation`
- `twist`
- `glide`

当多个行为事件重叠时，按配置中的 `classification.final_priority` 解决冲突。默认优先级：

```text
stagnation > twist > glide
```

## 6. PTZ 数据流程

PTZ 数据原始文件位于：

```text
data/PTZ/PTZ_*.csv
```

原始列包括：

```text
bodyparts_coords, Head_x, Head_y,
Trunk_x, Trunk_y, Tail_x, Tail_y, Z_y,
behavior.label
```

### 6.1 转换 PTZ 原始数据

运行：

```bash
python utils/convert_ptz_to_per_fish_raw.py --input-dir data/PTZ --output-dir data/raw/PTZ --scale-mm-per-pixel 0.30
```

输出：

```text
data/raw/PTZ/
  ptz_1_tracks_3d_interpolated.csv
  ptz_2_tracks_3d_interpolated.csv
  ...
  ptz_conversion_summary.csv

data/raw/
  ptz_tracks_3d_interpolated.csv
```

单鱼文件用于逐鱼批处理和排查；合并大表用于通用单文件 pipeline。

### 6.2 PTZ 批处理检测和自动评分

推荐使用 PTZ 专用批处理入口：

```bash
python src/PTZ/run_ptz_batch_detection.py --input-dir data/raw/PTZ --output-dir outputs/ptz_batch_classification
```

调用关系：

```text
src/PTZ/run_ptz_batch_detection.py
  -> discover_ptz_track_files()
  -> run_single_ptz_file()
      -> data_io.load_tracking_data()
      -> data_io.standardize_tracking_columns()
      -> preprocess.preprocess_tracking_data()
      -> features.compute_all_features()
      -> segmentation.segment_behavior()
      -> detection.detect_pause_candidates()
      -> classification.classify_behavior_candidates()
  -> merge_behavior_event_files()
  -> merge_feature_table_files()
  -> save_batch_run_summary()
  -> ptz_stagnation_accuracy.evaluate_ptz_stagnation()
```

输出结构：

```text
outputs/ptz_batch_classification/
  per_fish/
    ptz_1/
      final/behavior_events.csv
      intermediate/...
    ptz_2/
      final/behavior_events.csv
      intermediate/...
  merged/
    final/
      behavior_events.csv
    intermediate/features/
      feature_table.csv
    batch_run_summary.csv
  ptz_stagnation_accuracy/
    summary.json
    frame_metrics.csv
    frame_metrics_by_fish.csv
    event_metrics.csv
    event_matches.csv
    prediction_review.csv
    gt_im_events.csv
    pred_stagnation_events.csv
    unmatched_gt_events.csv
    unmatched_pred_events.csv
```

批处理脚本默认自动评分。如果只想跑检测和分类，不想评分：

```bash
python src/PTZ/run_ptz_batch_detection.py --input-dir data/raw/PTZ --output-dir outputs/ptz_batch_classification --no-evaluate
```

## 7. PTZ 专用评估口径

PTZ 专用评估只比较：

```text
prediction: stagnation
ground truth: IM
```

入口：

```bash
python src/PTZ/ptz_stagnation_accuracy.py --behavior-events outputs/ptz_batch_classification/merged/final/behavior_events.csv --feature-table outputs/ptz_batch_classification/merged/intermediate/features/feature_table.csv --output-dir outputs/ptz_batch_classification/ptz_stagnation_accuracy
```

评估分为两层。

帧级评估：

- 把预测的 `stagnation` 事件展开到逐帧。
- 把 `behavior_label == IM` 视为 GT positive。
- 空白未标注区域不作为完整负样本，统计为 `unknown_frames`。

事件级评估：

| 类别 | 含义 | 是否参与指标 |
| --- | --- | --- |
| `matched_im` | 预测 stagnation 与 IM 事件 IoU 达标 | 是，算 TP |
| `conflict_non_im` | 预测 stagnation 落在明确非 IM 标签上 | 是，算 FP |
| `unlabeled_candidate` | 预测只落在空白未标注区域 | 否，单独记录 |

这个口径适合 PTZ 数据，因为 PTZ 的行为标签不是完整逐帧真值；未标注区域不能简单当作确定的负样本。

## 8. 可视化

通用可视化入口：

```bash
python scripts/run_visualization.py --output-dir outputs/run_detection_example
```

PTZ 停滞事件诊断图入口：

```bash
python src/PTZ/ptz_stagnation_visualization.py --behavior-events outputs/ptz_batch_classification/merged/final/behavior_events.csv --feature-table outputs/ptz_batch_classification/merged/intermediate/features/feature_table.csv --output-dir outputs/ptz_batch_classification/ptz_stagnation_visualization --event-matches outputs/ptz_batch_classification/ptz_stagnation_accuracy/event_matches.csv
```

PTZ 可视化会围绕预测的 stagnation 事件截取窗口，展示三维轨迹、关键特征、标签区间和匹配信息，适合做误差分析。

## 9. 相对阈值模式

当前默认使用绝对阈值：

```yaml
adaptive:
  mode: absolute
```

如果希望基于当前数据分布生成相对阈值，可以运行：

```bash
python utils/generate_relative_thresholds.py --bootstrap-if-missing
```

它会读取 candidate segments，基于每条鱼的分布估计速度、窗口位移、运动能量、姿态稳定性等阈值，并写入：

```text
configs/adaptive/current_dataset_thresholds.yaml
```

之后把配置改为：

```yaml
adaptive:
  mode: relative
```

即可让 `load_config()` 自动叠加相对阈值。

## 10. 运行环境

项目代码使用 Python，并依赖常见的数据分析和科学计算库。当前仓库没有固定的环境锁文件，运行前需要确保环境中至少包含：

```
pandas, numpy, scipy, scikit-learn, pyyaml, matplotlib, openpyxl, pytest
```

如果使用 conda，可以先创建一个干净环境，再按本机习惯安装这些包。

## 11. 常用运行顺序

### PTZ 数据完整流程

```bash
python utils/convert_ptz_to_per_fish_raw.py --input-dir data/PTZ --output-dir data/raw/PTZ --scale-mm-per-pixel 0.30

python src/PTZ/run_ptz_batch_detection.py --input-dir data/raw/PTZ --output-dir outputs/ptz_batch_classification
```

可选生成诊断图：

```bash
python src/PTZ/ptz_stagnation_visualization.py --behavior-events outputs/ptz_batch_classification/merged/final/behavior_events.csv --feature-table outputs/ptz_batch_classification/merged/intermediate/features/feature_table.csv --output-dir outputs/ptz_batch_classification/ptz_stagnation_visualization --event-matches outputs/ptz_batch_classification/ptz_stagnation_accuracy/event_matches.csv
```

### 通用单文件流程

```bash
python scripts/run_detection.py --output-dir outputs/run_detection_example
python scripts/run_evaluation.py --output-dir outputs/run_detection_example
python scripts/run_visualization.py --output-dir outputs/run_detection_example
```

## 12. 输出文件说明

| 文件 | 说明 |
| --- | --- |
| `preprocessed_tracks.csv` | 添加时间戳、单位处理和平滑后的轨迹 |
| `feature_table.csv` | 每帧特征表，包含 `v`, `d_win`, `E_move`, `S_pose` 等 |
| `candidate_segments.csv` | 分段后的候选行为段 |
| `change_scores.csv` | 分段变点分数 |
| `candidate_segments_scored.csv` | 每个 segment 的 pause score |
| `candidate_segment_states.csv` | 每个 segment 的状态：`pause_candidate`, `transition`, `active` |
| `pause_candidate_events.csv` | 合并后的停滞候选事件 |
| `stagnation_events.csv` | 分类后的停滞事件 |
| `twist_events.csv` | 分类后的 twist 事件 |
| `glide_events.csv` | 分类后的 glide 事件 |
| `behavior_events.csv` | 最终行为事件表 |
| `batch_run_summary.csv` | PTZ 批处理中每条鱼的事件数量和输出路径 |
| `summary.json` | PTZ 专用评估摘要 |
| `prediction_review.csv` | PTZ 每个预测 stagnation 事件的匹配/冲突/未标注归类 |

## 13. 测试

运行核心测试：

```bash
python -m pytest tests/test_run_ptz_batch_detection.py tests/test_convert_ptz_to_per_fish_raw.py tests/test_ptz_data_io.py -q -p no:cacheprovider
```

运行全量测试：

```bash
python -m pytest -q -p no:cacheprovider
```

注意：当 `configs/default.yaml` 指向完整 PTZ 合并大表时，部分集成测试会比较慢。PTZ 全量实验推荐使用 `src/PTZ/run_ptz_batch_detection.py` 的逐鱼批处理入口。

## 14. 当前结果理解

PTZ 批处理可以跑通完整数据集并输出整体评分。最近一次全量批处理的结果显示，系统对 PTZ_1、PTZ_13 等鱼的 IM 停滞片段检测较好；全量 precision 被拉低的主要原因是部分鱼存在较长的 `TO` 或 `HYPE` 区间被规则误判为 `stagnation`。这说明当前系统更接近一个可解释的基础检测模块，后续提升方向主要是：

- 针对不同鱼或不同数据集引入相对阈值；
- 改进超长事件切分和事件合并策略；
- 对 `TO/HYPE` 等非 IM 低运动行为增加更细的区分规则；
- 增加更多可视化诊断和误差分析报告。
