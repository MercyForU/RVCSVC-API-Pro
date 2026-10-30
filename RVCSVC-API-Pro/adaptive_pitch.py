#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法:
  python adaptive_pitch.py vocals.mp3 --low A3 --high A5

F0 提取策略:
  优先 RMVPE (GPU/CPU)，失败自动回退 librosa.pyin (CPU)。

目录要求:
  adaptive_pitch.py 同级需有:
    rmvpe.pt
    infer/rmvpe.py（或 infer/lib/rmvpe.py）

RVCSVC-API 没带 RVC 的 tools/ 与 configs/ 包，本脚本会自动向
sys.modules 注入最小桩（tools.cuda_graph 的 passthrough、configs.config
的设备/dtype），不修改任何文件，RMVPE 即可正常走 GPU 推理。
"""

import os
import sys
import argparse
import numpy as np
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
_RMVPE = None
_RMVPE_FAILED = False


# ============================================================
# 音符 / 加权中位数 / 变调推荐
# ============================================================

def parse_note(x):
    if isinstance(x, str):
        return int(round(librosa.note_to_midi(x)))
    return int(round(float(x)))


def weighted_median(values, weights):
    """加权中位数：50% 权重落在它以下，50% 权重落在它以上。"""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    c = c / c[-1]
    return float(np.interp(0.5, c, v))


def recommend_transpose(midi, weights, comfort_low, comfort_high):
    c_low = parse_note(comfort_low)
    c_high = parse_note(comfort_high)
    if c_low > c_high:
        c_low, c_high = c_high, c_low
    c_center = (c_low + c_high) / 2.0

    song_median = weighted_median(midi, weights)
    t_float = c_center - song_median
    t_base = int(round(t_float))

    # 取整可能有两个候选，在 ±1 内选：
    # 1) 变调后加权中位数离舒适区中心最近
    # 2) 居中程度相同时，覆盖率更高
    # 3) 再相同，变调幅度更小
    best = None
    for t in (t_base - 1, t_base, t_base + 1):
        shifted = midi + t
        s_median = weighted_median(shifted, weights)
        center_pen = abs(s_median - c_center)
        inside = (shifted >= c_low) & (shifted <= c_high)
        coverage = float(weights[inside].sum() / weights.sum())

        key = (round(center_pen, 6), -coverage, abs(t - t_base))
        if best is None or key < best[0]:
            best = (key, t, coverage, s_median, center_pen)

    _, t, coverage, s_median, center_pen = best
    return {
        "transpose": t,
        "coverage": coverage,
        "song_median_midi": song_median,
        "shifted_median_midi": s_median,
        "comfort_center_midi": c_center,
        "center_pen": center_pen,
    }


# ============================================================
# RMVPE：GPU 快路径（只认同级目录）
# ============================================================

def _get_rmvpe(device=None):
    """懒加载并缓存 RMVPE 模型；只认同级 rmvpe.pt 和同级 infer/lib/rmvpe.py。"""
    global _RMVPE, _RMVPE_FAILED
    if _RMVPE is not None:
        return _RMVPE
    if _RMVPE_FAILED:
        return None

    model_path = os.path.join(_HERE, "rmvpe.pt")
    if not os.path.isfile(model_path):
        print(f"[RMVPE] 未找到同级 rmvpe.pt: {model_path}，回退 pyin")
        _RMVPE_FAILED = True
        return None

    # RMVPE 模块路径：优先 infer/rmvpe.py（RVCSVC-API 布局），回退 infer/lib/rmvpe.py
    rmvpe_module = None
    for cand in (
        os.path.join(_HERE, "infer", "rmvpe.py"),
        os.path.join(_HERE, "infer", "lib", "rmvpe.py"),
    ):
        if os.path.isfile(cand):
            rmvpe_module = cand
            break
    if rmvpe_module is None:
        print(f"[RMVPE] 未找到 infer/rmvpe.py 或 infer/lib/rmvpe.py，回退 pyin")
        _RMVPE_FAILED = True
        return None

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    import importlib.util

    try:
        import torch

        # ---- 注入缺失依赖的桩（不修改任何文件，只动 sys.modules）----
        # 1) tools.cuda_graph：infer/rmvpe.py 顶部 `from tools.cuda_graph import run_cuda_graph`
        #    RVCSVC-API 没有 RVC 的 tools/ 目录 → 注入 passthrough 桩（语义等价：在给定输入上执行 fn）
        if "tools.cuda_graph" not in sys.modules:
            import types
            _cuda_graph_stub = types.ModuleType("tools.cuda_graph")
            _cuda_graph_stub.run_cuda_graph = lambda module, name, fn, x, *a, **kw: fn(x, *a, **kw)
            _tools_pkg = sys.modules.get("tools") or types.ModuleType("tools")
            _tools_pkg.cuda_graph = _cuda_graph_stub
            sys.modules["tools"] = _tools_pkg
            sys.modules["tools.cuda_graph"] = _cuda_graph_stub

        _spec = importlib.util.spec_from_file_location("rmvpe_mod", rmvpe_module)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        RMVPE = _mod.RMVPE

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # 调用方可能传 torch.device 对象（app_rvc.py 的全局 device），统一成字符串
        device = str(device)
        want_cuda = device.startswith("cuda")
        if want_cuda and not torch.cuda.is_available():
            print("[RMVPE] 未检测到可用 CUDA，改走 CPU")
            device = "cpu"
            want_cuda = False

        # 2) configs.config：RMVPE.__init__ 在 device 为 "cuda*" 时会
        #    `from configs.config import get_device_dtype_sm`（RVC 的 configs 包），
        #    RVCSVC-API 没有 configs/ → 注入最小桩；我们固定 fp32，桩返回 fp32 即可。
        if want_cuda and "configs.config" not in sys.modules:
            import types
            _sel_device = torch.device(device)
            _sel_dtype = torch.float32
            _cfg_pkg = sys.modules.get("configs") or types.ModuleType("configs")
            _cfg_pkg.__path__ = []
            _cfg_mod = types.ModuleType("configs.config")
            _cfg_mod.get_device_dtype_sm = lambda idx=None, *a, **kw: (_sel_device, _sel_dtype, 0, 0)
            _cfg_mod.infer_device = _sel_device
            _cfg_mod.infer_dtype = _sel_dtype
            _cfg_pkg.config = _cfg_mod
            sys.modules["configs"] = _cfg_pkg
            sys.modules["configs.config"] = _cfg_mod

        # is_half=False：fp32 更稳，部分显卡 fp16 推理会出 NaN
        _RMVPE = RMVPE(model_path, is_half=False, device=device)
        # 缓存到全局后，同一进程内后续翻唱直接复用（不再重新加载/打印）
        print(f"[RMVPE] 已加载: {model_path} @ {_RMVPE.device}（fp32）")
        return _RMVPE
    except Exception as e:
        print(f"[RMVPE] 加载失败，回退 pyin: {e}")
        _RMVPE_FAILED = True
        return None


def _extract_f0_rmvpe(path, device=None, thred=0.03):
    """RMVPE 提取帧级 F0；失败返回 None。权重用全 1（按时间等权）。"""
    model = _get_rmvpe(device)
    if model is None:
        return None

    try:
        y, _ = librosa.load(path, sr=16000, mono=True)
        f0 = model.infer_from_audio(y, thred=thred)

        if hasattr(f0, "detach"):
            f0 = f0.detach().cpu().numpy()
        f0 = np.asarray(f0, dtype=np.float32)

        mask = np.isfinite(f0) & (f0 > 0)
        f0 = f0[mask]
        if len(f0) == 0:
            print("[RMVPE] 没有检测到有效人声帧，回退 pyin")
            return None

        midi = librosa.hz_to_midi(f0)
        weights = np.ones_like(midi, dtype=np.float32)
        return midi, weights
    except Exception as e:
        print(f"[RMVPE] 推理失败，回退 pyin: {e}")
        return None


# ============================================================
# pyin：CPU 回退路径
# ============================================================

def _extract_f0_pyin(mp3_path, fmin="C2", fmax="C6", sr=22050,
                     frame_length=2048, hop_length=512, prob_thr=0.3):
    """用 librosa.pyin 提取帧级 F0，权重用 voiced_prob，过滤静音和低置信帧。"""
    y, sr = librosa.load(mp3_path, sr=sr, mono=True)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    mask = voiced_flag & ~np.isnan(f0) & (voiced_prob >= prob_thr)
    f0 = f0[mask]
    prob = voiced_prob[mask]
    if len(f0) == 0:
        raise ValueError("没有检测到有效人声，请检查文件或降低 --prob-thr")
    midi = librosa.hz_to_midi(f0)
    return midi, prob


# ============================================================
# 统一入口
# ============================================================

def extract_f0(path, device=None, prefer_rmvpe=True, **pyin_kwargs):
    """
    RMVPE(GPU) 优先，失败自动回退 pyin(CPU)。
    返回 (midi: np.ndarray, weights: np.ndarray)。
    """
    if prefer_rmvpe:
        r = _extract_f0_rmvpe(path, device=device)
        if r is not None:
            return r
    return _extract_f0_pyin(path, **pyin_kwargs)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="根据纯人声和舒适区音域，计算 RVC 变调半音数"
    )
    parser.add_argument("audio", help="纯人声音频文件，mp3/wav 等")
    parser.add_argument("--low", required=True, help="舒适区最低音，如 A3")
    parser.add_argument("--high", required=True, help="舒适区最高音，如 A5")
    parser.add_argument("--fmin", default="C2", help="pyin 检测最低音，默认 C2")
    parser.add_argument("--fmax", default="C6", help="pyin 检测最高音，默认 C6")
    parser.add_argument("--prob-thr", type=float, default=0.3,
                        help="pyin 置信度阈值，默认 0.3")
    parser.add_argument("--no-rmvpe", action="store_true",
                        help="强制使用 pyin，不尝试 RMVPE")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出一个整数变调值")
    args = parser.parse_args()

    midi, weights = extract_f0(
        args.audio,
        prefer_rmvpe=not args.no_rmvpe,
        fmin=args.fmin,
        fmax=args.fmax,
        prob_thr=args.prob_thr,
    )
    r = recommend_transpose(midi, weights, args.low, args.high)

    if args.quiet:
        print(r["transpose"])
        return

    print(f"变调: {r['transpose']:+d} 半音")
    print(f"覆盖率: {r['coverage'] * 100:.1f}%")
    print(f"歌曲加权中位数: {librosa.midi_to_note(r['song_median_midi'])}")
    print(f"变调后加权中位数: {librosa.midi_to_note(r['shifted_median_midi'])}")
    print(f"舒适区中心: {librosa.midi_to_note(r['comfort_center_midi'])}")


if __name__ == "__main__":
    main()