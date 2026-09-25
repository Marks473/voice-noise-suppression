"""Этап 1: подготовка данных, контролируемое смешение речи и шума, ОСШ и VAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf

SAMPLERATE = 16000
EPS = 1e-12


def record_speech(duration_s: float, samplerate: int = SAMPLERATE) -> np.ndarray:
    import sounddevice as sd

    print(f"Идёт запись {duration_s:.1f} с — говорите после старта...")
    audio = sd.rec(int(duration_s * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("Запись завершена.")
    return audio.flatten()


def load_or_record_speech(path: Path, duration_s: float = 6.0, samplerate: int = SAMPLERATE) -> np.ndarray:
    path = Path(path)
    if path.exists():
        speech, sr = sf.read(path, dtype="float32")
        if sr != samplerate:
            raise ValueError(f"{path}: частота дискретизации {sr} Гц, ожидалась {samplerate} Гц")
        return speech
    path.parent.mkdir(parents=True, exist_ok=True)
    speech = record_speech(duration_s, samplerate)
    sf.write(path, speech, samplerate)
    print(f"Сохранено: {path}")
    return speech


def _normalize_rms(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2) + EPS)
    return x / rms


def synth_factory_noise(duration_s: float, samplerate: int = SAMPLERATE, seed: int | None = 0) -> np.ndarray:
    """Синтетический шум завода: гармоники двигателей + гул + случайные импульсы."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * samplerate)
    t = np.arange(n) / samplerate

    base_freq = 90.0
    harmonics = np.zeros(n)
    for k, amp in enumerate([1.0, 0.6, 0.35, 0.2], start=1):
        phase = rng.uniform(0, 2 * np.pi)
        harmonics += amp * np.sin(2 * np.pi * k * base_freq * t + phase)

    white = rng.standard_normal(n)
    alpha = 0.05
    broadband = sps.lfilter([alpha], [1, -(1 - alpha)], white)

    impulses = np.zeros(n)
    rate_hz = 0.7
    t_event = 0.0
    while True:
        t_event += rng.exponential(1.0 / rate_hz)
        if t_event >= duration_s:
            break
        idx = int(t_event * samplerate)
        decay_len = int(0.02 * samplerate)
        decay = np.exp(-np.arange(decay_len) / (0.003 * samplerate))
        amp = rng.uniform(1.5, 3.0)
        end = min(n, idx + decay_len)
        impulses[idx:end] += amp * decay[: end - idx]

    noise = 0.5 * _normalize_rms(harmonics) + 0.7 * _normalize_rms(broadband) + 0.3 * impulses
    return _normalize_rms(noise)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Смешивает речь и шум с заданным ОСШ (дБ). Возвращает (смесь, речь, шум) — согласованно
    отмасштабированные, если потребовалась защита от клиппинга."""
    n = len(speech)
    reps = int(np.ceil(n / len(noise)))
    noise_full = np.tile(noise, reps)[:n]

    p_speech = np.mean(speech**2) + EPS
    p_noise = np.mean(noise_full**2) + EPS
    k = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    noise_scaled = k * noise_full

    mixed = speech + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        speech = speech / peak
        noise_scaled = noise_scaled / peak
        mixed = mixed / peak
    return mixed, speech, noise_scaled


def _frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = max(1, 1 + (len(signal) - frame_len) // hop_len)
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = signal[start : start + frame_len]
    return frames


def estimate_windowed_snr(
    speech: np.ndarray, noise: np.ndarray, samplerate: int = SAMPLERATE, win_ms: float = 32.0, hop_ms: float = 16.0
) -> tuple[np.ndarray, np.ndarray]:
    """Кратковременная оценка ОСШ (дБ) по окнам. Возвращает (время центров окон, ОСШ)."""
    frame_len = int(win_ms / 1000 * samplerate)
    hop_len = int(hop_ms / 1000 * samplerate)
    speech_frames = _frame_signal(speech, frame_len, hop_len)
    noise_frames = _frame_signal(noise, frame_len, hop_len)

    p_speech = np.mean(speech_frames**2, axis=1) + EPS
    p_noise = np.mean(noise_frames**2, axis=1) + EPS
    snr_db = 10 * np.log10(p_speech / p_noise)

    centers = (np.arange(len(snr_db)) * hop_len + frame_len / 2) / samplerate
    return centers, snr_db


def energy_vad(
    signal: np.ndarray,
    samplerate: int = SAMPLERATE,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    threshold_offset_db: float = 6.0,
    floor_percentile: float = 10.0,
) -> list[dict]:
    """Энергетический VAD с фиксированным порогом. Возвращает таймлайн [{start, end, label}]."""
    frame_len = int(win_ms / 1000 * samplerate)
    hop_len = int(hop_ms / 1000 * samplerate)
    frames = _frame_signal(signal, frame_len, hop_len)
    energy_db = 10 * np.log10(np.mean(frames**2, axis=1) + EPS)

    noise_floor_db = np.percentile(energy_db, floor_percentile)
    threshold_db = noise_floor_db + threshold_offset_db
    is_speech = energy_db > threshold_db

    timeline = []
    frame_time = hop_len / samplerate
    start_idx = 0
    for i in range(1, len(is_speech) + 1):
        if i == len(is_speech) or is_speech[i] != is_speech[start_idx]:
            timeline.append(
                {
                    "start": round(start_idx * frame_time, 3),
                    "end": round(i * frame_time, 3),
                    "label": "speech" if is_speech[start_idx] else "silence",
                }
            )
            start_idx = i
    return timeline


def save_timeline(timeline: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Этап 1: данные, ОСШ, VAD")
    parser.add_argument("--speech", type=Path, default=Path("data/samples/speech.wav"))
    parser.add_argument("--snr", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    speech = load_or_record_speech(args.speech, args.duration)
    noise = synth_factory_noise(len(speech) / SAMPLERATE)
    mixed, speech_used, noise_used = mix(speech, noise, args.snr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_dir / "mix.wav", mixed, SAMPLERATE)

    overall_snr_db = 10 * np.log10(np.mean(speech_used**2) / (np.mean(noise_used**2) + EPS))
    print(f"Итоговый ОСШ смеси: {overall_snr_db:.2f} дБ (целевой: {args.snr:.2f} дБ)")

    _, snr_curve = estimate_windowed_snr(speech_used, noise_used)
    print(
        f"Кратковременный ОСШ: медиана {np.median(snr_curve):.1f} дБ, "
        f"мин {np.min(snr_curve):.1f} дБ (паузы), макс {np.max(snr_curve):.1f} дБ (активная речь)"
    )

    timeline = energy_vad(mixed)
    save_timeline(timeline, args.out_dir / "vad_timeline.json")
    print(f"VAD: {len(timeline)} сегментов, сохранено в {args.out_dir / 'vad_timeline.json'}")


if __name__ == "__main__":
    main()
