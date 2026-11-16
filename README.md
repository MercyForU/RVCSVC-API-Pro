# RVCSVC-API-Pro 🎤

基于 Gradio 的 AI 翻唱一体化服务：接收**音频下载直链**或**本地上传的音频文件**，自动完成「下载 → 人声分离（UVR5/MSST）→（可选）男/女声部提纯 → RVC/SVC 变声推理 → 专业音效后处理 → 混音导出 MP3」，并对外开放 Gradio API 供外部程序（如 AstrBot 插件）调用。

> 本项目默认使用自带的 `runtime/` 便携 Python（Windows 绿色版），无需另装环境。

---
## 部署方法
-   依据你的**显卡型号**选择合适的版本：[N卡50系（国内镜像）](https://hf-mirror.com/MercyStar/RVCSVC-API-Pro/resolve/main/rvc_cover_backend_50.7z?download=true)[10.9GB]； [N卡非50系（国内镜像）](https://hf-mirror.com/MercyStar/RVCSVC-API-Pro/resolve/main/rvc_cover_backend_N.7z?download=true)[11.0GB]； [A卡或核显（国内镜像）](https://hf-mirror.com/MercyStar/RVCSVC-API-Pro/resolve/main/rvc_cover_backend_A.7z?download=true)[8.82GB]；
- **依赖的模型（整合包已有）**：访问 [项目源RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 
## 目录

- [特性](#-特性)
- [项目结构](#-项目结构)
- [运行环境](#-运行环境)
- [前置依赖：先启动 RVC / SVC WebUI](#-前置依赖先启动-rvc--svc-webui)
- [模型文件放置规则](#-模型文件放置规则)
- [模型文件获取](#-模型文件获取)
- [启动服务](#-启动服务)
- [Web 界面使用](#-web-界面使用)
- [API 调用（供外部程序对接）](#-api-调用)
- [参数说明](#-参数说明)
- [双声道模式（男/女声部分离）](#-双声道模式男女声部分离)
- [自适应变调（RMVPE GPU 加速）](#-自适应变调rmvpe-gpu-加速)
- [清唱模式](#-清唱模式)
- [工作原理](#-工作原理)
- [音频后处理链](#-音频后处理链)
- [缓存机制](#-缓存机制)
- [常见问题](#-常见问题)
- [许可证 / 致谢](#-许可证)

---

## ✨ 特性

- 🎵 **双引擎**：RVC（`app_rvc.py`，端口 7860，代理 RVC WebUI 7897）与 SoVITS-SVC（`app_svc.py`，端口 7866，代理 so-vits-svc WebUI 7865）
- 🔗 **两种音源入口**：**本地上传音频文件**（mp3/wav/flac/m4a/ogg，也接受视频文件由 ffmpeg 抽音轨）或**音频下载直链**（支持第三方接口给出的 `music_url`）；上传优先于直链
- 🧩 **三种人声分离模型**：
  - `VR-HP5 (默认)` — UVR5 高速分离
  - `MSST BS-Roformer` — 高精度单模型分离
  - `MSST 串联管线 (becruily_deux + 去混响)` — 两步提干声 + 去混响
- 🎚️ **F0 音高提取算法可选**：`rmvpe`（默认，精度最高）/ `pm`（旧版）/ `fcpe`（全卷积轻量）
- 🎯 **自适应变调**：根据分离出的纯人声 F0 + 舒适区音域自动计算最优升降调（`±12` 钳制）；F0 提取**优先走 RMVPE GPU 快路径**（`adaptive_pitch.py` + 同级 `rmvpe.pt` + `infer/rmvpe.py`），失败自动回退 `librosa.pyin`（CPU）
- 👥 **双声道模式（男/女声部分离）**：合唱歌曲先提纯出男声、女声两条声部轨，**各自指定音色、各自自适应变调、各自走一次 RVC 推理**，再合成双声部 + 伴奏，实现「一人分饰男女两声」的合唱效果
- 🎤 **清唱模式**：推理完人声后不合成伴奏，直接输出纯人声（保留混响/延迟/压缩/响度归一化效果）
- 🎛️ **专业音效链**：EQ（高通/低通/三频 PeakFilter）+ 压缩 + 混响 + 延迟（延迟时间按伴奏 BPM 自动对齐）
- 🔁 **三级缓存**：分离结果（`output/`）、RVC 推理结果（`rvc_infer/`）、MF 男女分离结果（`mf/`）都会落盘复用，重跑同一首歌直接命中
- 🌐 **Web UI + Gradio API**：浏览器操作或外部程序 `/convert` 调用

---

## 📦 项目结构

```
RVCSVC-API-Pro/
├── app_rvc.py          # RVC 版主程序（直链 / 本地上传 / 双声道）
├── app_svc.py          # SoVITS-SVC 版主程序
├── msst_backend.py     # MSST 串联/BS-Roformer 分离后端
├── mf_separate.py      # 男/女声部分离（BS-Roformer Chorus Male/Female）
├── adaptive_pitch.py   # 自适应变调脚本（RMVPE GPU 快路径，可独立运行）
├── rmvpe.pt            # RMVPE 音高提取模型（GPU 快路径用）
├── infer/              # RVC 推理模块（rmvpe.py / fcpe.py / hubert.py / rtrvc.py ...）
├── uvr5/               # UVR5 分离模块
│   └── uvr_model/      # 分离模型文件（自动下载）
├── msst/               # MSST 模型权重与配置
│   ├── configs/        # 串联管线（becruily_deux / dereverb_echo）配置
│   └── pretrain/
│       ├── vocal_models/        # 人声/伴奏分离权重
│       ├── dereverb_models/     # 去混响权重
│       └── male_female_models/  # 男/女声部分离权重 + config（双声道用）
├── runtime/            # 自带 Python 便携运行时（Windows）
│   ├── python.exe
│   ├── ffmpeg.exe / ffprobe.exe
│   └── Lib/
├── output/             # 分离 / MF 分离 / RVC 推理缓存
├── temp/               # 最终输出 MP3、双声道合成中间文件
├── requirements.txt
├── start_rvcapi.bat    # RVC API 启动脚本（已带 NO_PROXY 代理绕过）
├── start_svcapi.bat    # SVC API 启动脚本
├── 使用说明.md         # 模型文件放置的简短备忘
└── 要使用svcapi请把文件夹里的文件替换到svc根目录/  # SVC API 对接补丁
```

> `adaptive_pitch.py`、`rmvpe.pt` 必须与 `app_rvc.py` **同级**（即 `RVCSVC-API-Pro/` 根目录），`infer/rmvpe.py` 在 `infer/` 目录下；缺失任一项时自适应变调自动回退 `librosa.pyin`（CPU），不阻断主流程。

---

## 🚀 运行环境

- **Windows**：直接用自带 `runtime\python.exe`，无需另装 Python
- **Linux/macOS**：Python 3.10+，CUDA 11.8+（有 GPU 时）
- 内存 ≥ 4 GB；跑双声道模式建议显存 ≥ 8 GB（MSST 串联管线 + MF 模型 + RVC 会先后占卡）

依赖（已包含在 `requirements.txt`）：

```
torch / torchaudio（按 GPU 选 cu118 或 cpu 版）
librosa, numpy, soundfile, resampy
gradio, requests, tqdm, pydub
pedalboard
```

---

## ⚠️ 前置依赖：先启动 RVC / SVC WebUI

**RVCSVC-API 本身不加载推理模型，而是代理到本地 WebUI 的推理端点。** 必须先启动对应 WebUI：

| 程序 | 代理目标 | 端点 |
|---|---|---|
| `app_rvc.py` | RVC WebUI | `http://127.0.0.1:7897`（`/run/infer_convert`、`/run/infer_change_voice`、`/run/infer_refresh`） |
| `app_svc.py` | so-vits-svc WebUI | `http://127.0.0.1:7865`（`/run/run_inference`） |

若你改了 WebUI 端口，修改对应 `app_*.py` 顶部的 `RVC_API_BASE` / `SVC_API_BASE`。

> **双声道模式的额外要求**：`app_rvc.py` 会把待推理的声部 wav 复制到 RVC WebUI 自己的 TEMP 目录，
> 路径写死为「与本项目**同级**、目录名为 **`RVC20260718`**」：
>
> ```
> RCV/rvc/
> ├── RVCSVC-API-Pro/     ← 本项目
> └── RVC20260718/        ← RVC WebUI 根目录（名字必须是这个，否则双声道推理读不到输入）
>     └── TEMP/
> ```
>
> 如果你的 RVC 目录名不同，改 `app_rvc.py` 中 `_infer_one_stem()` 里的 `rvc7897_temp` 变量即可。
> 普通（单轨）通道不受此限制。

### 使用 SVC API 的额外步骤

`so-vits-svc` 的 `app.py` 没有开放推理 API，需要把本项目里
`要使用svcapi请把文件夹里的文件替换到svc根目录/app.py` **覆盖**到 so-vits-svc 根目录，
并在 WebUI 中切换到「**独立目录模式**」。

---

## 📁 模型文件放置规则

> 原则：**文件直接放在指定目录，不要嵌套子文件夹；index/扩散/配置文件的命名要与模型文件名一致（去掉前缀）。**
> 所有模型权重（`.pth` / `.pt` / `.ckpt`）都不进 git 仓库，按下表各自获取后放到位。

### 模型文件获取

| 文件 | 放哪 | 获取来源 |
|---|---|---|
| RVC 的 `xxx.pth` | `RVC项目路径\assets\weights\` | 你自训的模型，或从 ModelScope/HuggingFace 下载 |
| RVC 的 `xxx.index` | `RVC项目路径\log\` | 与 `.pth` 同名（去 `added_IVF4949_Flat_nprobe_1_` 前缀） |
| `rmvpe.pt`（F0 提取，~173MB） | `RVCSVC-API-Pro\` 根目录 | 从 [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的 `rmvpe` 模型下载（或其 ModelScope 镜像）；没有它则自适应变调自动回退 CPU 的 `librosa.pyin`（不影响出结果，只是慢） |
| UVR5 `5_HP-Karaoke-UVR.pth` | `RVCSVC-API-Pro\uvr5\uvr_model\` | **首次启动自动下载**（ModelScope），无需手动 |
| MSST 模型（`.ckpt`/`.yaml`） | `RVCSVC-API-Pro\msst\pretrain\` | 仓库已含配置；权重放 `msst/pretrain/vocal_models\`、`msst/pretrain/dereverb_models\`；也可设环境变量 `PYMSS_STUDIO_DIR` 指向外部 Pymss-Studio 目录 |
| **男/女分离模型**（双声道用） | `RVCSVC-API-Pro\msst\pretrain\male_female_models\` | `Sucial/Chorus_Male_Female_BS_Roformer`：`model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt`（首选，SDR≈24.13）+ `config_chorus_male_female_bs_roformer.yaml`；缺 ep_267 会自动回退 `ep_146_sdr_23.8613.ckpt` |
| SVC 的 `xxx.pth`/`.yaml`/扩散 | `svc项目路径\models\` | 你自训的模型 |

### RVC（`app_rvc.py`）

| 文件 | 放置位置 | 命名 |
|---|---|---|
| `.pth` 模型 | `RVC项目路径\assets\weights\` | 与模型同名（如 `wuxiV1.pth`） |
| `.index` 索引 | `RVC项目路径\log\` | 与模型同名（如 `wuxiV1.index`） |

- 训练默认会生成 `added_IVF4949_Flat_nprobe_1_模型名.index`，请**把 `added_IVF4949_Flat_nprobe_1_` 前缀删掉**，只留与模型名一致的后缀。
- 配置文件 `.json` 同名放在 WebUI 可识别位置。

### 男/女声部分离（`mf_separate.py`，双声道专用）

| 文件 | 放置位置 | 命名 |
|---|---|---|
| `.ckpt` 权重 | `RVCSVC-API-Pro\msst\pretrain\male_female_models\` | `model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt` |
| `.yaml` 配置 | 同上 | `config_chorus_male_female_bs_roformer.yaml` |

> 该模型训练自 opencpop + m4singer 中文歌曲，**只能分离「男女同时唱」的合唱**；
> 男女交替（一段一段轮唱）时无法分离，输出会退化成整轨人声。
> 模型缺失/加载失败/显存不足时，`mf_separate` 返回空，`app_rvc.py` 自动回退为整轨单轨推理，不中断任务。

### SVC（`app_svc.py`）

| 文件 | 放置位置 | 命名 |
|---|---|---|
| `.pth` 模型 | `svc项目路径\models\` | 与模型同名 |
| 配置 `.yaml` | `svc项目路径\models\` | 与模型同名 |
| 扩散 `.pth` | `svc项目路径\models\diffusion\` | 与模型同名 |

模糊匹配阈值 40%（`threshold=0.4`），名字差别太大时匹配不到会报「未找到匹配的 index」。

---

## ▶️ 启动服务

```bash
# Windows（推荐，自带 runtime + 代理绕过）
start_rvcapi.bat

# 或直接调用 runtime
RVCSVC-API-Pro\runtime\python.exe RVCSVC-API-Pro\app_rvc.py

# Linux/macOS
python app_rvc.py          # RVC
python app_svc.py          # SVC
```

CPU 模式（无 GPU / 不想用 fp16）：

```bash
python app_rvc.py --is_nohalf
python app_svc.py --is_nohalf
```

首次启动会自动下载 UVR5 分离模型（约 120 MB），请耐心等待。

启动成功后访问：

- RVC：`http://127.0.0.1:7860`
- SVC：`http://127.0.0.1:7866`

> **代理绕过**：`start_rvcapi.bat` / `start_svcapi.bat` 已内置 `NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1`。若系统代理（Clash/V2Ray 等）拦截了 7860，Gradio 启动自检（`startup-events`）会 502 导致起不来——用这两个 `.bat` 启动即可绕过；若仍被系统级代理拦截，需在代理软件里把 `127.0.0.1` / `localhost` / `7860` 加入 TUN bypass。

---

## 🖥️ Web 界面使用

1. **刷新模型列表**：点 🔄「刷新」加载可用模型
2. **选择/加载模型**：下拉选模型，点 ✨「切换模型」（会同时自动匹配同名 `.index`）
3. **给音源**（二选一，**上传优先于直链**）：
   - 填写**音频下载直链**：一个可直接下载的 mp3/wav/flac/m4a URL
   - 或**上传音频文件**：mp3/wav/flac/m4a/ogg（也接受视频），以**文件名**作为缓存 key，重名可直接命中缓存
4. **选择分离模型（UVR）**：
   - `VR-HP5 (默认)`
   - `MSST BS-Roformer`
   - `MSST 串联管线 (becruily_deux + 去混响)`
5. **选择音高提取算法（F0）**：`rmvpe` / `pm` / `fcpe`
6. **自适应变调**：勾选「自适应变调」并填写舒适区低音/高音（如 A3 / A5）；若升降调为 0，后端会自动算最优升降调（`±12` 内钳制）
7. **清唱模式**：勾选「清唱模式（不合成伴奏）」，推理完只出纯人声
8. **双声道模式（男/女声部分离）**：勾选后强制改走 MSST 串联管线，并展开以下伴唱参数：
   - **伴唱音色**：男声轨用的 RVC 模型名（如 `maleV1.pth`），**留空 = 跟主唱同一音色**
   - **伴唱舒适区低/高音**：男声轨自适应变调的音域（默认 A3–A4；留空则跟主唱一致）
   - **伴唱音量 (dB)**：男声轨相对主唱的音量偏移，`-20 ~ +10`，默认 0，建议 −6（伴唱轻一点）
   - 主唱固定走**女声轨**、伴唱固定走**男声轨**
9. **调参**：
   - 升降调 `-12 ~ +12`（推荐女唱男 +6，男唱女 −6；伴奏会自动对齐）
   - 人声/伴奏音量 `-3 ~ +3` dB
   - 混响强度 `0 ~ 10`（0 干声，4 默认，10 宏大）
   - 回声（延迟）`0 ~ 10`（0 关闭）
10. **开始转换**：点 💕 按钮，等控制台打印「已导出」即完成，左侧播放器出结果

> 音频时长：超过 10 分钟会被自动截取前 10 分钟（UI 提示写的是 5 分钟，实际以代码 `600000ms` 为准）。

---

## 📡 API 调用

Gradio 暴露 `api_name`：`/convert`（RVC）与 `/convert`（SVC）、`/show_model`。

### RVC（`app_rvc.py`，端口 7860）

`/convert` 接受 **19 个参数**，顺序与 Web UI 一致：

```
[direct_link, key_shift, vocal_vol, inst_vol, model, reverb, delay,
 uvr_choice, f0_method, adaptive_shift, comfort_low, comfort_high,
 a_capella, dual_channel, upload_file, harmony_model,
 harmony_comfort_low, harmony_comfort_high, harmony_vol]
```

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
result = client.predict(
    "https://example.com/song.mp3",   # 1  direct_link 音频下载直链（用上传时传 None）
    0,                                 # 2  key_shift 升降调（半音，-12~+12）
    0,                                 # 3  vocal_vol 人声音量（dB）
    0,                                 # 4  inst_vol 伴奏音量（dB）
    "wuxiV1.pth",                      # 5  model RVC 模型名
    4,                                 # 6  reverb 混响强度（0~10）
    0,                                 # 7  delay 回声（0~10）
    "VR-HP5 (默认)",                  # 8  uvr_choice 分离模型（字符串全等）
    "rmvpe",                           # 9  f0_method 音高提取算法
    True,                              # 10 adaptive_shift 是否启用自适应变调
    "A3",                              # 11 comfort_low 舒适区低音
    "A5",                              # 12 comfort_high 舒适区高音
    False,                             # 13 a_capella 清唱模式（True=不合成伴奏）
    False,                             # 14 dual_channel 双声道模式（男/女声部分离）
    None,                              # 15 upload_file 本地上传文件（None=走直链）
    "",                                # 16 harmony_model 伴唱音色（""=跟主唱同音色）
    "A3",                              # 17 harmony_comfort_low 伴唱舒适区低音
    "A4",                              # 18 harmony_comfort_high 伴唱舒适区高音
    0,                                 # 19 harmony_vol 伴唱音量（dB，-20~+10）
    api_name="/convert",
)
print(result)   # 输出 MP3 的本地路径，需通过 /gradio_api/file= 下载
```

改用本地上传时，把第 1 个参数传 `None`、第 15 个参数传文件：

```python
result = client.predict(
    None, 0, 0, 0, "wuxiV1.pth", 4, 0,
    "MSST 串联管线 (becruily_deux + 去混响)", "rmvpe",
    True, "A3", "A5",
    False,              # a_capella
    True,               # dual_channel 开启双声道
    handle_file("D:/music/song.mp3"),   # 上传文件
    "maleV1.pth",       # 伴唱音色
    "A3", "A4", -6,     # 伴唱舒适区 + 音量
    api_name="/convert",
)
```

> ⚠️ `uvr_choice` 是**字符串全等**判断，必须原样写这三个值之一：
> `VR-HP5 (默认)` / `MSST BS-Roformer` / `MSST 串联管线 (becruily_deux + 去混响)`。
> 写错一个字符会被静默回退到 VR-HP5。
> ⚠️ `dual_channel=True` 时会**忽略** `uvr_choice`，强制使用 `MSST 串联管线 (becruily_deux + 去混响)`。
> ⚠️ 参数从旧版 13 个扩到 19 个，旧调用代码需要补齐后 6 个参数（否则 Gradio 会报参数数量不符）。

### SVC（`app_svc.py`，端口 7866）

SVC 的 `/convert` 参数较少（固定 F0 为 `fcpe`，无 UVR/F0/自适应/清唱/双声道选项）：

```python
client = Client("http://127.0.0.1:7866/")
result = client.predict(
    "https://example.com/song.mp3",
    0,
    "speaker0",                        # speaker_id
    api_name="/convert",
)
```

### 获取模型列表

```python
client = Client("http://127.0.0.1:7860/")
print(client.predict(api_name="/show_model"))
# ['wuxiV1.pth', 'guanguanV1.pth', ...]
```

---

## 🎚️ 参数说明

| 参数 | 取值 | 说明 |
|---|---|---|
| `direct_link` | http(s) URL | 音频下载直链，后端直接下载；**有上传文件时此参数被忽略** |
| `upload_file` | 文件路径 / FileData / None | 本地上传的音频，**优先于直链**；以文件名（去扩展名、清非法字符、截 80 字符）做缓存 key |
| `key_shift` | −12 ~ +12 | 人声/伴奏升降调（半音）；`>6` 或 `<−6` 会自动 `optimize_pitch_shift` 折叠 |
| `vocal_vol` / `inst_vol` | −3 ~ +3 | 人声/伴奏音量（dB） |
| `model` | 模型文件名 | RVC `.pth` 名（如 `wuxiV1.pth`），模糊匹配 index |
| `reverb` | 0 ~ 10 | 混响强度，0 干声 / 4 默认 / 10 宏大 |
| `delay` | 0 ~ 10 | 回声（延迟）强度，0 关闭；延迟时间按伴奏 BPM 算 `(60/BPM)×0.5` 秒（失败回退 0.5s），feedback 0.25，mix 最高 0.35 |
| `uvr_choice` | 三个固定字符串 | 见上，全等匹配；`dual_channel=True` 时被强制覆盖 |
| `f0_method` | `rmvpe`/`pm`/`fcpe` | 音高提取算法 |
| `adaptive_shift` | bool | 为 True 且 `key_shift==0` 时启用自适应变调 |
| `comfort_low` / `comfort_high` | 音名（如 A3/A5） | 自适应变调的舒适区音域（主唱 / 女声轨） |
| `a_capella` | bool | 清唱模式，True 时跳过伴奏合成 |
| `dual_channel` | bool | 双声道模式，True 时走「MF 男/女分离 → 各自推理 → 合成」 |
| `harmony_model` | 模型文件名 / 空串 | 双声道下**伴唱（男声轨）**的音色；空串 = 跟 `model` 同一音色 |
| `harmony_comfort_low` / `harmony_comfort_high` | 音名 / 空串 | 双声道下伴唱的自适应变调舒适区；空串 = 跟主唱一致 |
| `harmony_vol` | −20 ~ +10 | 双声道下伴唱轨相对主唱的音量偏移（dB），默认 0 |

---

## 👥 双声道模式（男/女声部分离）

勾选「双声道模式（男/女声部分离）」后，`convert()` 走下面这条链路（老通道逻辑完全不受影响）：

1. **强制换分离模型**：无论 UI 选了什么，都切到 `MSST 串联管线 (becruily_deux + 去混响)`。
   原因：MF 模型必须在「干声足够干净」时才能正确分男女，VR-HP5 的干声残留伴奏/混响偏多，喂进去会退化。
2. **MF 分离男女声部**：`mf_separate.separate_male_female()` 用 BS-Roformer Chorus Male/Female 模型把干声拆成 `male` / `female` 两条轨（写盘到 `output/{分离模型}/{song_key}/mf/`）。
   - 失败/模型缺失/显存不足 → **自动回退整轨单轨推理**（老通道），任务不中断
3. **主唱 = 女声轨，伴唱 = 男声轨**，两轨参数各自独立：

   | | 音色 | 舒适区 | 音量 |
   |---|---|---|---|
   | 主唱（女声轨） | `model` | `comfort_low`~`comfort_high` | 跟随 `vocal_vol` |
   | 伴唱（男声轨） | `harmony_model`（空=同主唱） | `harmony_comfort_low`~`harmony_comfort_high` | 额外 `harmony_vol` dB |

4. **保守串行推理**：`声部1 F0 → 声部1 RVC 推理 → 声部2 F0 → 声部2 RVC 推理`，避免两张卡/两次推理抢显存。
   每轨各自判断自适应变调（仅在 `key_shift==0` 且勾选自适应时生效），结果仍钳制在 `±12`。
5. **合成两声部**：截断到较短的采样数后相加，峰值归一化到 `0.95` 防削波，写成 `temp/{song_key}_mf_mixed.wav`。
6. **回到老通道后处理**：合成结果直接进 EQ/压缩/混响/延迟 + 响度归一化，再叠加伴奏导出。
   伴奏变调时用**伴唱（男声）的 shift** 作为代表值（伴奏通常跟低音区走）。
7. 清唱模式下同样只导出纯人声（此时即男女双声部合唱的干声）。

**限制与注意**：

- MF 模型只能分离**男女同时唱**的合唱；男女交替（一段一段轮唱）会退化成整轨人声，等于白跑一遍 MF
- 双声道整条链路需要：`mf_separate.py` + `msst/pretrain/male_female_models/` 下的 ckpt 与 yaml + MSST 串联管线权重
- RVC 目录需命名为 `RVC20260718` 且与本项目同级（见「前置依赖」里的说明），否则声部 wav 送不进 7897
- 双声道的结果缓存键包含声部 tag，主唱/伴唱不会互相串用缓存

---

## 🎯 自适应变调（RMVPE GPU 加速）

`app_rvc.py` 通过 `importlib` 动态加载**同级**的 `adaptive_pitch.py`：

1. 用 **RMVPE 模型**（`rmvpe.pt` + `infer/rmvpe.py`）提取分离后纯人声的帧级 F0
   - **GPU 快路径**：`RMVPE` 在 `cuda:0` 上跑 fp32，整首歌的人声 F0 提取比 pyin 快得多
   - **自动回退**：缺 `rmvpe.pt` / `infer/rmvpe.py` / GPU 不可用时，回退 `librosa.pyin`（CPU）
   - `adaptive_pitch.py` 会向 `sys.modules` 注入最小桩（`tools.cuda_graph` passthrough、`configs.config` 设备/dtype），不修改任何文件，使 RMVPE 模块在 RVCSVC-API 目录结构下也能正常加载
2. 计算加权中位数，对比舒适区中心（`(low+high)/2`）
3. 在 ±1 候选内按「变调后中位数离中心最近 → 覆盖率最高 → 幅度最小」选最优
4. 结果钳制在 `±12` 半音（RVC 允许范围）
5. 任何异常（文件不存在、加载失败、无有效 F0）都回退 `0`，不阻断主流程

> 双声道模式下，两条声部各自独立跑一次上述流程（`use_stem_direct=True`，直接对声部 wav 跑 F0，不走 UVR 缓存），所以主唱和伴唱可以拿到不同的变调值。

`adaptive_pitch.py` 也可独立运行：

```bash
python adaptive_pitch.py vocals.mp3 --low A3 --high A5
# 加 --quiet 只输出一个整数变调值
# 加 --no-rmvpe 强制走 pyin（不尝试 RMVPE GPU）
```

---

## 🎤 清唱模式

`a_capella=True` 时，`convert()` 在推理完人声、做完 EQ/压缩/混响/延迟/响度归一化后，
**跳过伴奏加载、伴奏变调、overlay 合成**，直接导出 `temp/{歌名}-清唱.mp3`（纯人声，保留所有音效）。
正常模式导出 `temp/{歌名}-RVC-AI翻唱.mp3`（人声+伴奏混音）。

---

## 🧱 工作原理

**普通（单轨）通道：**

```mermaid
graph LR
  A["音频直链 / 本地上传"] --> B[下载或读本地文件]
  B --> C[UVR5 / MSST 分离]
  C --> D[纯人声]
  C --> E[伴奏]
  D --> F[自适应变调? key_shift=0]
  F --> G[RVC/SVC 推理]
  G --> H[EQ/压缩/混响/延迟]
  H --> I{a_capella?}
  I -- 是 --> J[导出纯人声 MP3]
  I -- 否 --> K[伴奏变调对齐]
  K --> L[overlay 混音]
  L --> M[导出混音 MP3]
```

**双声道通道（`dual_channel=True`）：**

```mermaid
graph LR
  A["音频直链 / 本地上传"] --> B[强制 MSST 串联管线分离]
  B --> C[干净干声]
  B --> E[伴奏]
  C --> D1[MF 分离]
  D1 --> F1[女声轨 → 主唱音色]
  D1 --> F2[男声轨 → 伴唱音色]
  F1 --> G1[各自自适应变调 + RVC 推理]
  F2 --> G2[各自自适应变调 + RVC 推理]
  G1 --> H[两轨截断相加 / 峰值归一化 0.95]
  G2 --> H
  H --> I[EQ/压缩/混响/延迟]
  E --> J[按伴唱 shift 变调伴奏]
  I --> K{a_capella?}
  J --> K
  K -- 是 --> L[导出纯人声]
  K -- 否 --> M[overlay 混音 → 导出 MP3]
  D1 -. 失败回退 .-> N[整轨单轨推理]
  N --> I
```

---

## 🎛️ 音频后处理链（`pedalboard`）

```python
Pedalboard([
    HighpassFilter(80),
    PeakFilter(200, gain_db=1.5, q=0.7),
    PeakFilter(3000, gain_db=2.0, q=1.0),
    PeakFilter(7000, gain_db=-3.0, q=2.0),
    LowpassFilter(16000),
    Compressor(threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=150.0),
    # delay_intensity > 0 时追加 Delay：
    #   delay_seconds = (60 / 伴奏BPM) × 0.5（beat_track 失败则回退 0.5s）
    #   feedback=0.25, mix = (delay_intensity / 10) × 0.35
    Reverb(room_size=..., damping=0.4, wet_level=..., dry_level=..., width=0.8),
])
```

混响参数按 `reverb_intensity` 在 `0~4` / `4~10` 两段线性插值（`room_size` 0.15→0.40→0.90，`wet_level` 0.10→0.25→0.45，`dry_level = 1 − wet_level`）。响度归一化 `headroom=-1.0 dB`。

---

## 💾 缓存机制

三级缓存，全部落在 `output/` 下：

```
output/
├── UVR-HP5/{song_key}/
│   ├── vocal_{song_key}.wav_10.wav        # 人声
│   └── instrument_{song_key}.wav_10.wav   # 伴奏
└── MSST-BS-RoFormer/{song_key}/
    ├── {song_key}_vocals.wav              # 人声（串联管线：去混响后干声）
    ├── {song_key}_other.wav               # 伴奏
    ├── mf/                                # 双声道：男/女声部
    │   ├── {song_key}_vocals_male.wav
    │   └── {song_key}_vocals_female.wav
    └── rvc_infer/                         # RVC 推理结果
        └── {md5前12位}.wav
```

| 缓存 | 键 | 命中效果 |
|---|---|---|
| 分离缓存 | `song_key`（直链清洗后前 80 字符 / 上传文件名） | 跳过下载与 UVR/MSST 分离 |
| MF 分离缓存 | 干声文件名（`{song_key}_vocals`） | 跳过 MF 男/女分离推理 |
| RVC 推理缓存 | 单轨：`model + 实际shift + f0 + adaptive + comfort_low/high`<br>双声道：`声部tag + model + shift + f0` | 跳过 7897 推理，直接复用 wav |

- 缓存文件小于 44 字节（空 wav 头）视为无效，会重新计算
- **缓存键是直链本身**，直链变了（如 jkapi 换源）就会重新分离；想复用就改用**上传文件 + 同名**
- 想强制重跑：删掉对应 `{song_key}` 目录（或只删 `rvc_infer/` 保留分离结果）

---

## ❓ 常见问题

**Q: 提示「未能加载模型列表」**
A: 确认 RVC/SVC WebUI 已启动且端口正确（`RVC_API_BASE` / `SVC_API_BASE`）。

**Q: API 报「Value is not in the list of choices」**
A: `uvr_choice` / `f0_method` 等枚举参数传错。用「隐藏 API Convert」按钮对应的 `/convert`，参数顺序务必与 Web UI 一致；`uvr_choice` 要逐字符写对。

**Q: API 报参数数量不对 / 调用后行为错乱**
A: 新版 `/convert` 是 **19 个参数**（旧版 13 个）。老代码要按顺序补上 `dual_channel, upload_file, harmony_model, harmony_comfort_low, harmony_comfort_high, harmony_vol`，否则位置会整体错位。

**Q: 上传了文件但还是去下载直链 / 报「请填写音频下载直链」**
A: 上传框为空或路径不存在时才会回退直链。确认 `upload_file` 传的是真实存在的绝对路径（或 `handle_file()` 结果），且直链字段同时为空时会直接报错。

**Q: 双声道模式跑出来男女声没分开 / 只有一个人声**
A: ① 该模型只支持**男女同时唱**的合唱，轮唱分不开；② 确认 `msst/pretrain/male_female_models/` 下有 ckpt + `config_chorus_male_female_bs_roformer.yaml`；③ 看日志有没有 `[MF分离] 成功: male → ...`，出现 `回退整轨干声` 说明 MF 失败（多为显存不足或权重缺失）。

**Q: 双声道报「7897 推理失败」/ 读不到输入 wav**
A: `_infer_one_stem()` 会把声部 wav 复制到 `../RVC20260718/TEMP/`。确认 RVC WebUI 目录名为 `RVC20260718` 且与本项目同级，或改掉 `app_rvc.py` 里的 `rvc7897_temp` 路径。

**Q: 音频质量不佳**
A: 调小 `key_shift`（避免跨度过大）、提高模型训练质量、或换 `rmvpe` F0 算法。

**Q: 直链下载失败**
A: 确认直链可匿名访问（VIP 歌曲加密直链通常不行）；检查网络。也可以直接上传本地文件绕过。

**Q: Gradio 启动自检 502（startup-events 报错）**
A: 系统代理拦截了 7860。用 `start_rvcapi.bat`（已带 `NO_PROXY` 绕过），或在代理软件里把 `localhost` / `127.0.0.1` / `7860` 加 TUN bypass。

**Q: 自适应变调没生效 / 走的是 pyin 不是 RMVPE**
A: 确认 `adaptive_pitch.py`、`rmvpe.pt` 在 `RVCSVC-API-Pro/` 根目录且 `infer/rmvpe.py` 存在；且 `adaptive_shift=True` 且 `key_shift==0` 才会触发。看后端日志有没有 `[RMVPE] 已加载: ... @ cuda:0`——没有就是回退到了 pyin（缺文件或缺 GPU）。

**Q: 改了参数/新增功能但没生效**
A: 后端需重启才能识别新的 UI 参数与代码改动；同时清一下浏览器缓存（Gradio 前端配置会缓存）。

---

## 📄 许可证

本项目仅供学习、研究和个人娱乐使用。请勿用于商业用途或生成侵害他人权益的内容；使用者对生成内容自行承担法律责任。

## 🙏 致谢

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — RVC 模型
- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) — SVC 模型
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) — 人声分离
- [Gradio](https://github.com/gradio-app/gradio) — Web 界面框架
- 项目作者 [CCYellowStar](https://www.modelscope.cn/models/CCYellowStar/RVCSVC-API)
- 项目贡献者 [ABCwewe](https://github.com/ABCwewe/RVCSVC-API)
