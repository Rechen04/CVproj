# CoDe-GS：稀疏视角 3D Gaussian Splatting 的一致性加权深度先验改进

## 当前服务器运行入口（2026-11-09）

- 服务器地址与部署路径记录在本地 `training/deployment.json`；当前部署的项目目录为 `/chenpx/CoDeGS`。
- 独立环境、临时文件和缓存：`/chenpx/codegs_setup`；每次运行先执行 `source scripts/server_env.sh`。
- 完整流程：`python scripts/run_managed.py --jobs 4`，依次执行 fern 短实验、30 组调参、112 组主实验与消融、图表和视频。
- 运行状态：`training/status.json`；安装与自检日志：`training/bootstrap.log`；各阶段日志：`training/smoke.log`、`training/sweep.log`、`training/main.log`、`training/assets.log`。
- 调参候选按 fern/horns/room 上平均 PSNR 与 LPIPS(vgg) 的名次和排序，PSNR 用于同分排序；选中配置写入 `training/frozen_parameters.json`，正式实验统一使用相同步数和参数。
- 实验结果：`results/SUMMARY.md`、`results/SUMMARY.csv`、各场景模型与渲染；展示文件：`assets/`。
- 部署信息与验证记录：`training/deployment.json`、`training/environment_validation.json`。当前使用 PyTorch 2.4.1+cu121、gsplat 1.5.3，调用服务器已有 CUDA/cuDNN 运行库；深度模型以代码推理使用的 FP16 精度保存。
- LLFF 数据采用 [NeRF Baselines 数据副本](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/tree/main/llff)，保留 8 个场景的 `images_8`、`sparse` 和 `poses_bounds.npy`。

重新启动时，可用 `--start sweep`、`--start main` 或 `--start assets` 选择阶段；主实验会跳过已完整保存模型、指标和点云的实验。

> 课程项目：改进原始论文 **3D Gaussian Splatting for Real-Time Radiance Field Rendering**（Kerbl et al., SIGGRAPH 2023 / ToG，最佳论文）。
> 只给 **3 张照片** 时原版 3DGS 严重过拟合（漂浮物、几何破碎）；本项目在其上加入单目深度大模型先验，并用**跨视角一致性**判断先验可不可信。

## 1. 故事线（汇报用）

1. **原论文**：3DGS 用上百万个各向异性高斯 + 可微光栅化，实现实时、高质量的新视角合成；但它依赖**密集视角**（一个场景几十到几百张图）。
2. **发现的问题**：LLFF 只用 3 张训练图时，3DGS 的 PSNR 从密集视角的 ~26 dB 掉到 ~15–17 dB。DNGaussian（CVPR'24）表 1 中，3DGS 的 3-view 结果是 15.52 / SSIM 0.408 / LPIPS 0.405。新视角下会出现大量漂浮物，深度图也是碎的。根本原因是**光度损失在 3 个视角下欠约束**：只要训练视角拟合好，几何可以是错的。
3. **改进（CoDe-GS）**：三个模块，每个都能单独消融。
   - **DInit：深度对齐的稠密初始化。** 用 Depth-Anything-V2 预测每张训练图的相对视差，再用训练视角可三角化的 SfM 点做鲁棒仿射拟合，把它对齐到真实尺度。然后反投影成稠密点云来初始化高斯，替代原版那几百个稀疏 SfM 点。
   - **CW-D：一致性加权的局部 + 全局 Pearson 深度先验。** 单目深度在一些区域是错的，比如反光、细小结构、远景。把视角 i 的对齐深度反投影到 3D，再投到视角 j，和 j 自己的对齐深度比较，算出跨视角一致性置信度 conf ∈ [floor, 1]，计算时考虑遮挡。尺度不变的 Pearson 深度损失按这个置信度加权：相信"多视角互相印证"的深度，弱化不一致的区域。
   - **PV：伪视角正则。** 在训练相机之间插值（旋转用 slerp）并加小扰动，生成未见视角。渲染后在线跑 DAv2，用同样的 Pearson 损失约束伪视角的几何，直接压制"只在新视角才暴露"的漂浮物。
4. **实验**：LLFF 8 个场景，3/6/9 视角，标准稀疏视角协议（每第 8 张做测试，其余用 linspace 取 n 张训练，图像下采样 8×）。完整消融做 8 组，指标是 PSNR / SSIM / LPIPS，加上高斯数、训练时间、FPS。
5. **展示**：3DGS 和 Ours 左右对比的环绕飞行视频，上面 RGB、下面深度。外加定性网格图、学习曲线、先验与置信度可视化，以及 `.ply` 导出到 SuperSplat 网页里现场拖动 3D。

**诚实定位**：单目深度先验（FSGS、SparseGS、DNGaussian）、伪视角（FSGS）、深度可靠性加权（如 *In Depth We Trust*, arXiv 2604.05715）都已有工作。本项目的贡献是在原版 3DGS 上**以统一框架组合，加入几何一致性置信度，并做完整消融**。汇报时请这样表述，不要说"首次提出"。

## 2. 目录

```
train.py                 训练 + 测试评估（--preset 选方法）
render_video.py          环绕视频；多个 ckpt 自动左右对比（上 RGB / 下深度）
sparsegs/
  data.py                LLFF/COLMAP 读取 + 稀疏视角划分
  colmap_io.py           COLMAP 二进制读写
  depth_prior.py         DAv2 推理、尺度对齐、跨视角一致性置信度、反投影
  losses.py              L1+D-SSIM、加权全局/局部 Pearson、平滑项
  render.py              gsplat 光栅化封装（另带纯 PyTorch CPU 版用于自测）
  cameras.py             伪视角采样、螺旋相机路径
  init.py / utils.py     高斯初始化；LPIPS 指标、PLY 导出、深度上色
scripts/
  run_smoke.sh           第 1 天：环境 + fern 两个短实验
  run_sweep.sh           第 2 天：3 个开发场景上小范围调参
  run_llff.sh            第 3–4 天：8 场景 × 3/6/9 视角 主表 + 消融（可断点续跑、可并行）
  make_assets.sh         第 5 天：所有对比视频 / 网格图 / 曲线 / 先验图
  summarize.py           汇总成 Markdown + CSV 表
  make_figure.py / plot_curves.py / precompute_depth.py
offline/                 离线服务器打包与安装（见 offline/README_OFFLINE.md）
tests/make_synthetic_scene.py   合成小场景，用于无 GPU 时自测整条流程
```

## 3. 方法预设（`--preset`）

| preset | 初始化 | 深度损失 | 伪视角 | 含义 |
|---|---|---|---|---|
| `3dgs_random` | 随机 | – | – | 原版 3DGS，随机初始化（DNGaussian 表中的设定） |
| `3dgs` | 训练视角 SfM | – | – | **原论文基线** |
| `depth_sfm` | SfM | Pearson（均匀） | – | 已有工作风格的深度正则（强基线） |
| `dinit` | 深度反投影 | – | – | 消融：只换初始化 |
| `dinit_depth` | 深度 | Pearson（均匀） | – | 消融 |
| `dinit_conf` | 深度 | Pearson（一致性加权） | – | 消融 |
| `full_noconf` | 深度 | Pearson（均匀） | ✓ | 消融：去掉置信度 |
| `full` | 深度 | Pearson（一致性加权） | ✓ | **CoDe-GS（Ours）** |

## 4. 快速开始

```bash
# 离线服务器：先按 offline/README_OFFLINE.md 装好，然后每次：
source ~/miniconda3_codegs/bin/activate
cd CoDeGS
bash scripts/run_smoke.sh                      # ~3 分钟：环境自检 + fern 上 3dgs vs full
JOBS=4 bash scripts/run_sweep.sh               # 调参（fern/horns/room）
EXTRA="--w_pseudo 0.05" JOBS=4 bash scripts/run_llff.sh   # 主实验（EXTRA 填调参选出的值）
bash scripts/make_assets.sh                    # 视频和图
cat results/SUMMARY.md
```

单个实验：

```bash
python train.py --data_dir data/nerf_llff_data/fern --preset full --n_views 3
python render_video.py --ckpts results/fern/3dgs_3v/ckpt.pt results/fern/full_3v/ckpt.pt \
       --labels "3DGS" "Ours" --out assets/videos/fern.mp4
```

每个 run 的输出在 `results/<scene>/<preset>_<n>v/`，包括：`metrics.json`（总体和逐视角指标、学习曲线、先验对齐统计）、`test/`（测试渲染、GT、深度图）、`prior/`（单目深度与置信度图）、`ckpt.pt`、`point_cloud.ply` 和 `log.jsonl`。

**无 GPU 时自测**（这套流程我已经在 CPU 上跑通）：

```bash
python tests/make_synthetic_scene.py --out data/synthetic/toy --n 16 --w 48 --h 36
python train.py --data_dir data/synthetic/toy --preset full --backend torch --device cpu --steps 200 \
       --refine_start 50 --refine_every 50 --pseudo_start 100 --pseudo_every 5 --depth_patch 16 \
       --depth_model weights/Depth-Anything-V2-Small-hf
```

## 5. 七天计划（A100 一张，离线）

| 天 | 目标 | 产出 / 检查点 |
|---|---|---|
| D0（今天） | 在有网的电脑上运行 `offline/download_online.py`，拷贝到服务器，执行 `install_offline.sh` | 自检脚本最后打印 `All set` |
| D1 | `run_smoke.sh`；确认 fern 上 `3dgs` 的 PSNR 在 15–18 范围，`full` 明显更高；看 `prior/` 里的深度和置信度图是否合理 | 管线可用。如果 `full` 不比 `3dgs` 好，先查对齐 corr（日志 `depth prior:` 里的 corr 应 > 0.5） |
| D2 | `run_sweep.sh`（约 1–2 小时），按 PSNR+LPIPS 选 `w_depth`、`conf_tau`、`w_pseudo`、步数 | 冻结超参，写进 `EXTRA` |
| D3 | `run_llff.sh`：3 视角全部 8 组 × 8 场景，然后 6/9 视角 | `results/SUMMARY.md` 主表 + 消融表 |
| D4 | 补实验和分析：失败场景、置信度覆盖率、高斯数/速度对比；可选 Mip-NeRF360 24 视角 | 讨论素材 |
| D5 | `make_assets.sh`：挑 2–3 个最惊艳的场景（通常 fern / horns / trex / room），导出 PLY 到 SuperSplat 录屏 | 视频、图 |
| D6 | 写 PPT + 报告；按"原论文 → 问题 → 方法 → 实验 → 消融 → 局限"组织 | 交付 |
| D7 | 缓冲 / 排练 | |

## 6. 已知注意事项

- **相机畸变**：LLFF 自带的 COLMAP 相机是 SIMPLE_RADIAL，所以原版 3DGS 代码会直接报错。本仓库只读取焦距和主点、忽略很小的径向畸变，和 NeRF/RegNeRF/DNGaussian 的 LLFF 协议一致，因此不需要跑 COLMAP 去畸变。
- **gsplat 1.5.3 的 DefaultStrategy 有运算符优先级 bug**：`step % reset_every == 0 & step > 0` 永远为 False，所以不透明度重置不会发生。稀疏视角下我们本来就默认不重置；需要时用 `--reset_every N`，由本仓库自己实现。
- **SfM 点来源**：LLFF 自带的 `sparse/0` 是用全部视角重建的。我们只保留"能被 ≥2 个训练视角观测到"的点，用于初始化和尺度对齐，这是常见近似（FSGS 是用训练视角重新跑 COLMAP）。报告里要写明。
- **调参只在 3 个开发场景上做**，然后在 8 个场景上统一报告，报告里要写明。
- 和论文数字对比时要注意：DNGaussian 用随机初始化，并把 SH 换成了神经颜色渲染器；FSGS 训练 10k 步。我们自己的 `3dgs`、`3dgs_random` 是同一代码、同一预算下的公平对照，主结论以它们为准。
