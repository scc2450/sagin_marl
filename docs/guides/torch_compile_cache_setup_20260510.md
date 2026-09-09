# torch.compile 缓存路径设置

## 为什么需要显式设置

当前训练主线会使用 `torch.compile` / Inductor / Triton 来加速 critic 和部分 actor 更新。Windows 上默认缓存通常落在：

```text
C:\Users\<user>\AppData\Local\Temp\torchinductor_<user>
```

这个默认位置有两个问题：

- 缓存会占用 C 盘空间。
- 如果缓存路径包含中文或其他非 ASCII 字符，Triton/Inductor 在 Windows 上可能出现解码或编译缓存错误。

所以项目约定：**训练前显式设置 ASCII 缓存路径，推荐放在 D 盘。**

## 推荐 PowerShell 设置

当前推荐路径：

```powershell
$env:TORCHINDUCTOR_CACHE_DIR = "D:\sagin_cache\torchinductor"
$env:TRITON_CACHE_DIR = "D:\sagin_cache\triton"
```

这些变量必须在启动 Python 训练命令之前设置。已经运行中的 Python 进程不会自动切换缓存目录。

## 写入用户环境变量

如果希望以后新开的 PowerShell / IDE 终端自动继承，可以写入用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable("TORCHINDUCTOR_CACHE_DIR", "D:\sagin_cache\torchinductor", "User")
[Environment]::SetEnvironmentVariable("TRITON_CACHE_DIR", "D:\sagin_cache\triton", "User")
```

写入后通常需要重新打开终端或 IDE 终端。当前已经打开的终端仍建议手动设置一次 `$env:...`。

## 训练入口检查

训练入口会在 CUDA + compile 相关功能启用时打印：

```text
[torch-compile-cache:<entry>] TORCHINDUCTOR_CACHE_DIR=... TRITON_CACHE_DIR=... torchinductor_cache_dir=...
```

如果变量没设置、路径在 C 盘、或路径包含非 ASCII 字符，会给出 warning。这个检查不改变训练语义，也不会在代码里自动重定向缓存。

## 验证命令

```powershell
Get-ChildItem Env:TORCHINDUCTOR_CACHE_DIR
Get-ChildItem Env:TRITON_CACHE_DIR

.\.venv\Scripts\python.exe -c "import os, torch._inductor.codecache as cc; print(os.environ.get('TORCHINDUCTOR_CACHE_DIR')); print(os.environ.get('TRITON_CACHE_DIR')); print(cc.cache_dir())"
```

## 不建议

不要把缓存目录放到当前仓库路径下，例如：

```text
D:\研三上\毕设\sagin_marl\...
```

这类路径包含非 ASCII 字符，可能触发 Triton cache 解码错误。
