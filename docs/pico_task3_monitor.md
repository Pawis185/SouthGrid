# Pico Task3 本地评分与原始场景日志

## 当前交付状态

新增功能默认关闭。开启后，采集端逐物理子步复制真实本地 MuJoCo 状态，独立日志进程落盘，独立主机监控进程调用**安装包中未修改的官方 Task3 方法**。不构造 ScoringEngine，不创建 attempt，不连接 leaderboard。

已完成离线验证和合成监控页面检查；**本版本尚未通过真实 ORCA + Pico + 双相机端到端验收**。2026-09-15 检查时 50051、8001 未监听。本功能不能为旧人工录像补造场景证据。

## 启动

正常准备 OrcaLab 场景、双相机和 Pico 后，在接受本版本的仓库目录运行：

```bash
conda activate orcalab_lerobot
cd /home/ubuntu/Documents/Codex/2026-09-15/southgrid-pico-score-logging/work/SouthGridNew/src/examples/dataCollection/g1_omnipicker
adb reverse tcp:8001 tcp:8001
python g1_omnipicker_collection_tele_lerobot.py \
  --task_config ../common/example.yaml \
  --lerobot_out ~/datasets/g1_tele \
  --repo_id local/g1_omnipicker \
  --task 整理工具 \
  --fps 20 --clock wall \
  --cameras head,wrist_r --cam_resolution 480x640 \
  --camera_source websocket --task3_monitor
```

主机浏览器：**http://127.0.0.1:8766/**。端口冲突用 `--task3_monitor_port 8767`。服务仅监听本机。Pico 头显没有新增画面通路，需要采集者能看到主机屏幕；页面为世界坐标俯视/侧视示意图，不是相机投影。

续采追加 `--resume`。本仓库当前分类实现原本就按 `good/qualified/Doubtful` 自动追加；`--resume` 兼容保留。scene_sidecar 使用 UUID，从不复用分类 episode index。旧 scratch 和旧数据保持原状。

去掉 `--task3_monitor` 即运行原有采集流程，不启动场景日志、评分进程或物理步包装。该入口另提前初始化 ORCA 官方 logger 的 `log_dir`，并为 Numba 缓存设置可写位置，避免安装目录只读导致导入失败；不修改安装包。

监控模式目前要求 websocket；`mp4 + --task3_monitor` 在解析参数时明确报错，避免假装掌握 MP4 曝光关联。原有不带监控的 MP4 模式仍保留。

## 操作与读数

当前源码的操作是：左 Grip 开始 → 左 Grip 结束 → 松键 → 左 Grip=好、X=合格、A=存疑、右 Grip=删除。录制中右 Grip 放弃，双 Grip 退出。底盘控制原本关闭，本改动没有开启它：reach 区域可见不代表此入口能操纵底盘到目标站位。

- 显示总分 /40、reach /5 和五工具各 /7，以及官方诊断。`pick_voltage_pen` 明确映射 **ElectriciansKnife02 电工刀02**，不是测电笔。
- 录制中为完整已落盘**轨迹前缀**评分；短时惩罚会随时长变化，分数不保证单调。录制结束待分类阶段保留前缀结果，并注明最终日志校验待提交。分类提交后，自动后台读取原始文件，检查哈希/连续性再导出结束复核。
- 缺字段、数据断流、日志缺口、时钟跳变或评分版本不匹配时显示不可评分；前缀超过 5 秒显示过期，不用 0 代替缺失。评分进程按约 2 秒间隔计算，页面每 0.5 秒读取；计算变慢时显示实际延迟。
- 使用官方 `FrameData.ts=time.time()` 同类的主机墙钟时间依据；另独立记录真实 MuJoCo `data.time` 和主机单调时钟。未把仿真时间或固定帧率补算时间冒充主机时间。
- 调用官方 `_auto_score_all_steps` 和 `_apply_task3_penalties` 等原方法（运行时从只读安装源码 AST 提取，不重写公式）。使用官方 FrameOracle、task chain 和 predicates。**不含基于正式 attempt_id 的末尾确定性 jitter**，因此是本地规则分，不是 leaderboard 成绩。官方 FrameOracle 自身对部分运动查询有固定行为，适配不改动它。
- 抓取/释放展示为接触与夹爪原始读数的辅助观察；“无机器人接触”不声称已完成可靠抓取/释放。位置稳定辅助阈值与官方评分条件分开显示。

## 对齐标记

页面提供全场、箱内 10 cm、峰值 5 mm 视野。所有单位为米，矩阵行主序，四元数 wxyz。

- 理论工具峰值中心为工具箱 site；近似满分半径 1.249 mm，并标出持续门槛半径 0.3 m。reach 为三维距离的薄球壳；俯视图画的是该球壳与当前底盘高度平面的交线，不能将其误读为固定 0.65 m 水平圆。
- 五工具当前位置、已观察抓取末端点/轴、释放工具点分开显示；局部点以 `world = site_position + site_rotation @ local_point` 跟随真实对象姿态。
- 抓取/释放局部参考提取自已有真实 `original_candidate_00100/replay.json.gz`，保存来源哈希；只读取既有冻结区域文件，未搜索新点、未修改前四工具策略。冻结文件原始 tool 编号与用户顺序不同，始终按 step_id/site 映射。
- theoretical / observed / recommended 原始范围和来源在展开栏中保留。observed_box_relative 的原来源含世界轴差值，未冒称经过物体旋转验证的局部鲁棒区域。换布局/换姿态后的抓取与释放标记是**参考外推，未做新的实机验证**；上方 10 cm 虚线是未验证接近提示，不保证避碰。
- 不创建碰撞体，不改 site/body/布局。原始训练相机像素不经过绘制。独立 SVG 示意预览自动写 `previews/`；页面按钮可另外导出 PNG 到浏览器下载目录。

## 日志和离线复核

```text
~/datasets/g1_tele/
  good/ | qualified/ | Doubtful/     原有分类 LeRobot
  scene_sidecar/
    current.json                    当前 UUID
    service.log
    collection.lock                 防止两个采集进程共用 sidecar
    episodes/<UUID>/
      manifest.json                 版本、任务/对象/相机映射、分类和实际 episode index
      scene.mjb                     当时加载的真实模型及其哈希
      raw.jsonl                     原始 physics/control/camera/lifecycle 记录（不覆盖）
      raw_digest.json               原始文件 SHA256 和字节数
      latest.json                   派生的最新场景快照
      score_snapshots.jsonl         派生的前缀评分历史
      score_latest.json             派生的最新评分
      review.json                   派生的结束复核或不可评分原因
      review_worker.log
      previews/                     独立标记 SVG，非训练图像
```

原始证据与派生文件有独立名称/子目录；复核不会改写 raw.jsonl。每条原始记录有连续 record_id 和双主机时钟；每个 physics 有连续 sample_id、control_id、真实 sim_time、qpos/qvel/act、实际 ctrl、全部 site 姿态、机器人/工具/箱体位置与姿态/速度、全部关节状态、每子步接触对/距离/接触点/坐标轴。scene.mjb 提供几何及关节索引解释。每控制步另记录实际命令和 Pico 原始按键状态。

`camera_capture` 保存相机源 frame index、取帧前后单调时间区间、对应 physics/control ID、重复帧和跳过的源帧数量。`lerobot_frame` 明确连接“上一次图像/状态”和“本次目标状态”：原实现 LeRobot action 是下一采样状态，真实执行器命令在 physics/control_applied 中。**现有 CameraWrapper 没有提供可核验的曝光时间戳**，该字段为 null，不声称实现曝光级同步；相机缺失或采样跳帧仍可影响训练可用性，必须另行验收。

记录 reset 完成、录制开始/结束、状态变化、放弃/删除/保存/异常退出，以及超过 250 ms 的主机采样停顿。未知外部暂停原因不伪装成已观测 pause 指令。未开始时以约 5 Hz 保存 idle_preview 供对齐查看，不进入评分轨迹，避免等待时间污染 episode 时长。

队列有上限，采集不等待磁盘/评分。日志逐行写，约 0.5 秒 fsync，集末 fsync 并校验。队列溢出、写进程异常使最终评分失效。断电可能丢失最近尚未 fsync/尚在队列中的数据，不能保证零损失；open manifest 在续采时标记 `interrupted_before_resume`，保留完整行和残尾，绝不补写历史数据。异常/未提交轨迹不输出完整 episode 分。

采集程序退出后监控服务关闭；结束复核 worker 会完成文件导出。需要重新打开页面可运行：

```bash
python -m task3_monitor.service --root ~/datasets/g1_tele/scene_sidecar --port 8766
```

单集离线复核（同一脚本目录、同一 conda 环境，不连接 ORCA）：

```bash
python -m task3_monitor.service --review ~/datasets/g1_tele/scene_sidecar/episodes/实际UUID
```

安装的评分源码/任务配置变化后，不自动认可旧哈希：需对新版本重新回归适配及理论区域。不要修改 `scoring_version.json` 来绕过检查。

## 验证记录与待验

通过：6 项离线 unittest，覆盖真实 MuJoCo 玩具模型逐子步读取/模型序列化、批量与逐步物理等价/瞬时接触、坐标变换和五工具映射、相机动作对应/重复帧、唯一 UUID/完整写入/断尾/篡改/缺字段/时钟跳变拒绝、同轨迹前缀与结束复核一致。玩具模型全部明确标为 synthetic_test，不是 ORCA/Pico 成功证据。

已有真实场景 replay 1291 帧，用原始时间/位置/速度回归官方方法得到 40.0（reach=5、五工具=7），与既有结果一致。这不替代本次采集链路实机验收。原参数 `--help` 在实际 conda 环境可运行；监控关闭时控制与存储调用路径保持原样，但真实不开启监控的采集仍未在本轮重跑。

浏览器验证了合成数据的页面渲染、六项评分、五工具选项、接触摘要和毫秒级时间/毫米缩放控件。没有真实 camera exposure 精度测试，也没有证明目标工作站的 20 Hz 持续帧率、控制延迟、长期磁盘吞吐或 Pico 内可见性。首次实机请用独立新数据目录，核查 source/model/scene 对应、classification index、camera index、日志完整性、延迟和结束复核结果；若 journal overflow 或延迟持续过大，该集不可作为完整评分证据。

OFFICIAL_ATTEMPTS_SENT=0
