import os
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import librosa
import soundfile as sf
import torch


MSST_ROOT = Path(__file__).resolve().parent / "msst"
MSST_SELECTION = "MSST BS-Roformer"
MSST_SELECTIONS = ["MSST BS-Roformer", "MSST 串联管线 (becruily_deux + 去混响)"]

# 外部 Pymss-Studio 模型目录（可选回退，不依赖固定绝对路径）：
# 1. 优先读环境变量 PYMSS_STUDIO_DIR（部署时按需设置，如 D:\Tools\Ai\Audio\Pymss-Studio-...）
# 2. 否则用旧默认（D 盘）——仅当该目录真实存在时才启用，避免其它机器上硬失败
def _default_pymss_studio_dir() -> Path:
    env = os.environ.get("PYMSS_STUDIO_DIR")
    if env:
        return Path(env)
    legacy = Path("Pymss-Studio/data/models")
    return legacy if legacy.exists() else Path(".")

PYMSS_STUDIO_DIR = _default_pymss_studio_dir()

MODEL_PROFILES = {
    # 步骤 1 模型: becruily_deux (原始输入 -> Vocals, Instrumental)
    "becruily_deux": {
        "model_type": "mel_band_roformer",
        "relative_model": Path("pretrain/vocal_models/becruily_deux.ckpt"),
        "relative_config": Path("configs/vocal_models/becruily_deux.yaml"),
        "external_model": PYMSS_STUDIO_DIR / "vocal/vocal_instrumental_dual/becruily_deux.ckpt",
        "external_config": PYMSS_STUDIO_DIR / "vocal/vocal_instrumental_dual/becruily_deux.yaml",
    },
    # 步骤 2 模型: dereverb_echo (步骤 1 Vocals -> dry, other)
    "dereverb_echo": {
        "model_type": "mel_band_roformer",
        "relative_model": Path("pretrain/dereverb_models/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt"),
        "relative_config": Path("configs/dereverb_models/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.yaml"),
        "external_model": PYMSS_STUDIO_DIR / "reverb_echo_control/dereverb_deecho/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt",
        "external_config": PYMSS_STUDIO_DIR / "reverb_echo_control/dereverb_deecho/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.yaml",
    },
    # 兼容原单模型配置
    "single_bs_roformer": {
        "model_type": "bs_roformer",
        "relative_model": Path("pretrain/vocal_models/bs_roformer_karaoke_frazer_becruily.ckpt"),
        "relative_config": Path("configs/vocal_models/config_karaoke_frazer_becruily.yaml"),
        "external_model": None,
        "external_config": None,
    },
}

if str(MSST_ROOT) not in sys.path:
    sys.path.insert(0, str(MSST_ROOT))

from inference.msst_infer import MSSeparator


_SEPARATOR_CACHE: dict[tuple[str, str], MSSeparator] = {}
_SEPARATOR_LOCK = threading.RLock()


def resolve_model_asset(model_key: str, root: Path = MSST_ROOT) -> tuple[str, Path, Path]:
    if model_key not in MODEL_PROFILES:
        raise ValueError(f"Unknown model profile key: {model_key}")
    profile = MODEL_PROFILES[model_key]
    model_type = profile["model_type"]

    # 1. 优先使用项目目录下的模型和配置
    local_model = root / profile["relative_model"]
    local_config = root / profile["relative_config"]
    if local_model.is_file() and local_config.is_file():
        return model_type, local_model, local_config

    # 2. 回退到 Pymss-Studio 目录
    ext_model = profile.get("external_model")
    ext_config = profile.get("external_config")
    if ext_model and ext_config and ext_model.is_file() and ext_config.is_file():
        return model_type, ext_model, ext_config

    # 若未找到，抛出友好提示
    raise FileNotFoundError(
        f"未找到模型文件或配置文件: {model_key}\n"
        f"期望路径: {local_model} 或 {ext_model}"
    )


def get_separator(root: Path, model_key: str, device: str, output_dir: Path) -> MSSeparator:
    root = Path(root).resolve()
    cache_key = (model_key, device)
    with _SEPARATOR_LOCK:
        separator = _SEPARATOR_CACHE.get(cache_key)
        if separator is not None:
            return separator

        model_type, model_path, config_path = resolve_model_asset(model_key, root)
        separator = MSSeparator(
            model_type=model_type,
            config_path=str(config_path),
            model_path=str(model_path),
            device=device,
            output_format="wav",
            store_dirs=str(output_dir),
            inference_params={"batch_size": None, "num_overlap": None, "chunk_size": None, "normalize": None},
        )
        _SEPARATOR_CACHE[cache_key] = separator
        return separator


def separate_msst_audio(
    input_path: str | Path,
    output_dir: str | Path,
    root: Path = MSST_ROOT,
    model_selection: str = MSST_SELECTION,
) -> tuple[str, str]:
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"MSST input audio not found: {input_path}")

    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(">>> 启动 MSST 两阶段串联人声分离管线:")
    print("   [步骤 1] 原始输入 -> becruily_deux.ckpt (提取 Vocals 与 Instrumental)")
    print("   [步骤 2] 步骤 1 Vocals -> dereverb_echo.ckpt (提取纯净干声 dry)")
    print("=" * 65)

    mix, sample_rate = librosa.load(str(input_path), sr=44100, mono=False)

    # ========== 步骤 1: becruily_deux ==========
    sep1 = get_separator(root, "becruily_deux", device, output_dir)
    with _SEPARATOR_LOCK:
        t0 = time.perf_counter()
        results1 = sep1.separate(mix)
        print(f"[OK] 步骤 1 (becruily_deux) 分离完成，耗时: {time.perf_counter() - t0:.1f}s")

    norm1 = {str(k).lower(): v for k, v in results1.items()}
    vocal_audio = norm1.get("vocals")
    instrumental_audio = norm1.get("instrumental")
    if instrumental_audio is None:
        instrumental_audio = norm1.get("other")

    if vocal_audio is None or instrumental_audio is None:
        raise RuntimeError("步骤 1 (becruily_deux) 未能成功分离出 Vocals 和 Instrumental")

    # ========== 步骤 2: dereverb_echo ==========
    # 保证送入步骤 2 的音频是 (channels, samples) 形状
    if vocal_audio.ndim == 2 and vocal_audio.shape[0] > vocal_audio.shape[1]:
        vocal_input_step2 = vocal_audio.T
    else:
        vocal_input_step2 = vocal_audio

    sep2 = get_separator(root, "dereverb_echo", device, output_dir)
    with _SEPARATOR_LOCK:
        t1 = time.perf_counter()
        results2 = sep2.separate(vocal_input_step2)
        print(f"[OK] 步骤 2 (dereverb_echo) 去混响与回声完成，耗时: {time.perf_counter() - t1:.1f}s")

    norm2 = {str(k).lower(): v for k, v in results2.items()}
    dry_vocal_audio = norm2.get("dry")
    if dry_vocal_audio is None:
        print("[WARN] 步骤 2 未找到 'dry' 音轨，回退使用步骤 1 原始人声")
        dry_vocal_audio = vocal_audio

    # ========== 写入输出文件 ==========
    vocal_path = output_dir / f"{input_path.stem}_vocals.wav"
    instrumental_path = output_dir / f"{input_path.stem}_other.wav"
    raw_vocal_path = output_dir / f"{input_path.stem}_vocals_raw.wav"

    sf.write(vocal_path, dry_vocal_audio, sample_rate, subtype="FLOAT")
    sf.write(instrumental_path, instrumental_audio, sample_rate, subtype="FLOAT")
    sf.write(raw_vocal_path, vocal_audio, sample_rate, subtype="FLOAT")

    print(f"[DONE] 串联管线处理完成！已生成:")
    print(f"     * 纯净干声人声 (送入转换): {vocal_path}")
    print(f"     * 伴奏音轨 (用于混音): {instrumental_path}")
    print(f"     * (备份) 步骤1未去混响人声: {raw_vocal_path}")

    return str(vocal_path), str(instrumental_path)
