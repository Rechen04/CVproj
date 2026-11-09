# 离线服务器部署指南

服务器不能上网，所以分两步：**① 在有网的电脑上打包 → ② 拷到服务器上安装**。环境固定为 **Python 3.10 + PyTorch 2.4.1 + gsplat 1.5.3 预编译 wheel**。这套组合不需要 nvcc，也不需要编译 CUDA。

## 0. 先在服务器上看一眼（30 秒）

把 `offline/check_server.sh` 拷到服务器上运行：

```bash
bash check_server.sh
```

脚本最后一行会推荐 CUDA 版本：驱动 ≥550 用 `cu124`，≥530 用 `cu121`，更老的用 `cu118`。torch 的 wheel 自带 CUDA 运行库，只要求驱动版本够新。

## 1. 有网的电脑上打包（Windows / macOS / Linux 都行，Python ≥ 3.8）

```bash
cd CoDeGS
python offline/download_online.py --cuda cu124          # 国外网络
python offline/download_online.py --cuda cu124 --cn     # 国内：清华 PyPI + hf-mirror + 清华 Miniconda
```

输出目录是 `offline_bundle/`，大约 6–7 GB：

| 内容 | 来源 |
|---|---|
| `Miniconda3-py310-Linux-x86_64.sh` | anaconda / 清华镜像 |
| `wheels/`：torch 2.4.1、torchvision 0.19.1、13 个 nvidia-*/triton CUDA 库、gsplat 1.5.3+pt24cuXXX、transformers、lpips 等 | PyTorch 官方源、PyPI、GitHub Release |
| `weights/Depth-Anything-V2-{Large,Small}-hf/` | HuggingFace（或 hf-mirror） |
| `torch_hub_checkpoints/`：vgg16、alexnet | LPIPS 依赖的 torchvision 权重 |
| `nerf_llff_data.zip`（1.8 GB） | NeRF 官方 Google Drive，通过 gdown 下载 |
| `CoDeGS/` + `install_offline.sh` | 本仓库 |

**跨平台下载的坑（脚本已处理）**：在 Windows/Mac 上用 `pip download --platform linux` 时，pip 会**静默跳过** torch 依赖的 nvidia-* 和 triton，因为这些依赖的条件按本机判断。脚本会从 torch wheel 的 METADATA 里读出这 13 个包，再单独下载。

**LLFF 下载失败时**（Google Drive 在国内通常要代理），任选一个手动下载 `nerf_llff_data.zip`，放到 `offline_bundle/` 下：

- NeRF 官方 Google Drive 文件夹：https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 （文件 id `16VnMcF1KJYxN9QId6TClMsZRahHNMW5g`）
- HuggingFace 数据集 `poptree/LLFF`：需登录并申请访问（manual gated），里面有 `nerf_llff_data.zip`
- 同学或课题组如果已有 NeRF 数据，直接拷贝 `nerf_llff_data/`（每个场景需包含 `images_8/` 和 `sparse/0/`）

可以用 `--skip` 跳过已有的部分，比如 `--skip llff miniconda`。脚本可以重复运行，已下载的文件会跳过。

## 2. 拷到服务器并安装

```bash
scp -r offline_bundle user@server:~/       # 或者用 U 盘、跳板机
ssh user@server
cd ~/offline_bundle && bash install_offline.sh
```

安装脚本会在 `~/miniconda3_codegs` 装一个**独立的** Python 3.10，不影响服务器上已有的环境，然后离线安装所有 wheel，把权重放到位、解压 LLFF，最后自检：CUDA 光栅化、DAv2 加载、LPIPS。看到 `All set` 就可以开始了：

```bash
source ~/miniconda3_codegs/bin/activate
cd ~/offline_bundle/CoDeGS
bash scripts/run_smoke.sh
```

## 常见问题

- **`torch.cuda.is_available()` 为 False**：驱动太旧。重新用 `--cuda cu121` 或 `--cuda cu118` 打包 wheels（加 `--skip miniconda models lpips llff repo`，只重下 wheels）。
- **找不到 `libcudnn` / `libcublas`**：wheels 里缺 nvidia-* 包。检查 `wheels/` 里是否有 13 个 `nvidia_*` 和 `triton` 文件。
- **长时间运行断网/断 SSH**：用 `tmux` 或 `nohup bash scripts/run_llff.sh > run.log 2>&1 &`。`run_llff.sh` 支持断点续跑：已有 `metrics.json` 的 run 会跳过。
- **LPIPS 报下载错误**：检查 `~/.cache/torch/hub/checkpoints/` 里有没有 `vgg16-397923af.pth` 和 `alexnet-owt-7be5be79.pth`。
