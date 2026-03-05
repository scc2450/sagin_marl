# 租算力训练全流程（Windows 本地 -> 云端 Linux）

本文按你当前项目给出可直接执行的流程，覆盖：
- 选型（CPU/GPU 怎么选）
- 连接服务器
- 环境初始化
- 训练
- 训练过程查看
- 评估
- 结果保存与下载
- 关机止损

## 1. 先回答你的问题：CPU 还是 GPU？

结论：这个项目不建议只堆 CPU，推荐“单卡 GPU + 足够 CPU 核”。

原因（来自项目代码）：
- 环境采样主要吃 CPU：`scripts/train.py` 支持 `--num_envs` + `--vec_backend subproc` 多进程并行。
- 训练与评估会自动优先用 GPU：`sagin_marl/rl/mappo.py`、`scripts/evaluate.py` 都是 `torch.cuda.is_available()` 为真就走 CUDA。
- 当前网络规模不大（Actor/Critic 均为 2 层 `256` 隐层），通常单卡就够，CPU 负责把 rollout 吞吐拉起来。

推荐租机档位：

| 场景 | 推荐配置 | 说明 |
| --- | --- | --- |
| 调参/冒烟 | 8 vCPU, 16-32 GB RAM, 可无 GPU | 可跑，但速度慢 |
| 日常训练（推荐） | 12-16 vCPU, 32 GB RAM, 1 张中档 GPU（>=12 GB 显存） | 性价比通常最好 |
| 高并发实验 | 24+ vCPU, 64 GB RAM, 1-2 张 GPU | 多实验并行更稳 |

补充：
- 你的项目当前更像“CPU 采样 + GPU 训练”的混合负载，不是纯 GPU 训练。
- 如果预算有限，宁可保留 1 张普通 GPU，也不要只买很多 CPU 核。

## 2. 租机前检查清单

下单前确认：
- 是否可 SSH 登录（有公网 IP、端口、用户名、密码/密钥）。
- 是否带 NVIDIA 驱动（GPU 机型）。
- 系统版本建议 Ubuntu 22.04/24.04。
- 系统盘至少 50 GB（训练日志 + checkpoint + TensorBoard）。
- 计费方式是否支持“关机不计费/低计费”。
- 是否支持快照（训练完成后可封存环境）。

## 3. 从 Windows 连接服务器

在本地 PowerShell：

```powershell
ssh -p <PORT> <USER>@<SERVER_IP>
```

首次连接会提示指纹，输入 `yes`。

建议先生成 SSH 密钥并上传（免密登录）：

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

把公钥追加到服务器 `~/.ssh/authorized_keys`。

## 4. 服务器初始化

登录服务器后执行：

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip tmux htop
```

如果是 GPU 机，先确认驱动：

```bash
nvidia-smi
```

## 5. 上传或拉取项目代码

优先使用 Git（推荐）：

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone <你的仓库地址> sagin_marl
cd sagin_marl
```

如果没有远程仓库，可以从本地上传（在本地 PowerShell 执行）：

```powershell
scp -P <PORT> -r . <USER>@<SERVER_IP>:~/workspace/sagin_marl
```

## 6. Python 环境与依赖安装

在服务器项目根目录：

```bash
cd ~/workspace/sagin_marl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证 PyTorch/GPU：

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

建议先跑一个冒烟测试：

```bash
python -m pytest tests/test_smoke_train.py -q
```

## 7. 训练（建议用 tmux）

新建会话，避免断线中断训练：

```bash
tmux new -s sagin_train
```

### 7.1 Stage 1（加速度）

```bash
python scripts/train.py \
  --config configs/phase1_actions_curriculum_stage1_accel.yaml \
  --log_dir runs/phase1_actions \
  --run_id stage1_accel \
  --num_envs 8 \
  --vec_backend subproc \
  --torch_threads 8 \
  --updates 400
```

### 7.2 Stage 2（带宽头，继承 Stage 1）

```bash
python scripts/train.py \
  --config configs/phase1_actions_curriculum_stage2_bw.yaml \
  --log_dir runs/phase1_actions \
  --run_id stage2_bw \
  --init_actor runs/phase1_actions/stage1_accel/actor.pt \
  --init_critic runs/phase1_actions/stage1_accel/critic.pt \
  --num_envs 8 \
  --vec_backend subproc \
  --torch_threads 8 \
  --updates 500
```

### 7.3 Stage 3（卫星头，继承 Stage 2）

```bash
python scripts/train.py \
  --config configs/phase1_actions_curriculum_stage3_sat.yaml \
  --log_dir runs/phase1_actions \
  --run_id stage3_sat \
  --init_actor runs/phase1_actions/stage2_bw/actor.pt \
  --init_critic runs/phase1_actions/stage2_bw/critic.pt \
  --num_envs 8 \
  --vec_backend subproc \
  --torch_threads 8 \
  --updates 500
```

参数建议（按 CPU 核数调整）：
- 8 vCPU：`--num_envs 4 --torch_threads 4`
- 16 vCPU：`--num_envs 8 --torch_threads 8`
- 24+ vCPU：从 `--num_envs 8` 逐步加到 `12` 或 `16`，实测吞吐后再定

tmux 常用：
- 退出不停止：`Ctrl+b` 然后按 `d`
- 重新进入：`tmux attach -t sagin_train`

## 8. 训练过程查看

### 8.1 看训练产物

- checkpoint：`runs/phase1_actions/<RUN_ID>/actor.pt`、`critic.pt`
- 训练指标：`runs/phase1_actions/<RUN_ID>/metrics.csv`

```bash
ls -lah runs/phase1_actions/<RUN_ID>
tail -n 5 runs/phase1_actions/<RUN_ID>/metrics.csv
```

### 8.2 快速分析指标

```bash
python scripts/analyze_metrics.py --run_dir runs/phase1_actions/<RUN_ID> --window 20
```

### 8.3 TensorBoard（本地浏览器查看）

服务器上启动：

```bash
tensorboard --logdir runs/phase1_actions/<RUN_ID> --host 127.0.0.1 --port 6006
```

本地 PowerShell 开隧道：

```powershell
ssh -p <PORT> -L 16006:127.0.0.1:6006 <USER>@<SERVER_IP>
```

浏览器打开：

```text
http://127.0.0.1:16006
```

## 9. 评估（训练后）

在服务器项目根目录执行。

评估训练策略：

```bash
python scripts/evaluate.py \
  --config configs/phase1_actions_curriculum_stage1_accel.yaml \
  --run_dir runs/phase1_actions/<RUN_ID> \
  --episodes 20
```

评估启发式基线（推荐一起跑）：

```bash
python scripts/evaluate.py \
  --config configs/phase1_actions_curriculum_stage1_accel.yaml \
  --run_dir runs/phase1_actions/<RUN_ID> \
  --episodes 20 \
  --baseline queue_aware
```

混合评估（accel 用训练策略，bw/sat 用启发式）：

```bash
python scripts/evaluate.py \
  --config configs/phase1_actions_curriculum_stage1_accel.yaml \
  --run_dir runs/phase1_actions/<RUN_ID> \
  --episodes 20 \
  --hybrid_bw_sat queue_aware
```

评估输出：
- `runs/phase1_actions/<RUN_ID>/eval_trained.csv`
- `runs/phase1_actions/<RUN_ID>/eval_baseline.csv`
- `runs/phase1_actions/<RUN_ID>/eval_tb/`

## 10. 渲染与结果导出

生成单条轨迹 GIF：

```bash
python scripts/render_episode.py \
  --config configs/phase1_actions_curriculum_stage1_accel.yaml \
  --run_dir runs/phase1_actions/<RUN_ID>
```

常见结果文件：
- `actor.pt`, `critic.pt`
- `metrics.csv`
- `eval_trained.csv`, `eval_baseline.csv`
- `eval_tb/`

## 11. 结果保存与回传（重点）

先在服务器压缩：

```bash
cd ~/workspace/sagin_marl/runs/phase1_actions
tar -czf <RUN_ID>.tar.gz <RUN_ID>
```

再从本地 PowerShell 下载：

```powershell
scp -P <PORT> <USER>@<SERVER_IP>:~/workspace/sagin_marl/runs/phase1_actions/<RUN_ID>.tar.gz .
```
如：
打开 AutoDL 的控制台，找到你那个实例的 “SSH联系方式”，通常长这样：ssh -p 34567 root@region-1.autodl.com
# 注意：-P 是大写的
scp -P 34567 root@region-1.autodl.com:~/workspace/sagin_marl/runs/phase1_actions/stage1_accel.tar.gz .

建议长期保存两份：
- 本地硬盘
- 对象存储/网盘

## 12. 结束训练与关机止损

确认训练进程：

```bash
ps -ef | grep "scripts/train.py" | grep -v grep
```

需要停止时：

```bash
kill <PID>
```

确认结果已打包下载后再关机：

```bash
sudo shutdown -h now
```

## 13. 常见问题排查

1. `torch.cuda.is_available()` 是 `False`
- 先看 `nvidia-smi` 是否正常。
- 再确认当前 Python 环境里安装的是可用的 PyTorch 版本。

2. CPU 很满但训练慢
- 先降 `--num_envs`，再调 `--torch_threads`，避免过度抢核。
- 优先观察 `metrics.csv` 中 `env_steps_per_sec`、`update_steps_per_sec`。

3. 断网后训练停了
- 必须用 `tmux` 或 `nohup` 跑长任务。

4. 找不到结果文件
- 训练命令是否用了 `--run_id` 或 `--run_dir`。
- 每次训练先记录终端输出的 run 目录。

5. 判断机器运行状态
- htop判断多核状态
- nvidia-smi判断GPU利用率

## 14. 最短闭环命令（可直接复用）

```bash
source .venv/bin/activate
python scripts/train.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --log_dir runs/phase1_actions --run_id exp_cloud_001 --num_envs 8 --vec_backend subproc --torch_threads 8 --updates 400
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir runs/phase1_actions/exp_cloud_001 --episodes 20
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir runs/phase1_actions/exp_cloud_001 --episodes 20 --baseline queue_aware
python scripts/analyze_metrics.py --run_dir runs/phase1_actions/exp_cloud_001 --window 20
```

