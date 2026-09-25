"""Этап 1: подготовка данных, контролируемое смешение речи и шума, ОСШ и VAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    """Приводит сигнал к единичной среднеквадратичной мощности: x / RMS(x), где
    RMS(x) = sqrt(mean(x**2)). После этого мощность сигнала равна ровно 1, и его
    дальше можно предсказуемо смешивать с другими сигналами произвольной исходной
    громкости — см. mix().
    """
    rms = np.sqrt(np.mean(x**2) + EPS)
    return x / rms


def synth_factory_noise(duration_s: float, samplerate: int = SAMPLERATE, seed: int | None = 0) -> np.ndarray:
    """Синтетический шум завода = гул моторов + широкополосный шум.

    Модель — сумма двух слагаемых, каждое приведено к единичной мощности и взято
    с своим весом:

        w(t) = 0.6 * (гул) + 0.4 * (широкополосный шум)

    Гул — это сумма нескольких гармоник основной частоты f0 (кратные частоты
    f0, 2f0, 3f0, ... возникают из-за периодического вращения вала двигателя или
    вентилятора; чем выше гармоника, тем меньше её амплитуда — так задан список
    [1.0, 0.6, 0.35, 0.2]). Широкополосный шум — обычный белый гауссовский шум
    (независимые нормальные случайные числа), он моделирует всё остальное:
    трение, вибрацию корпуса, электрические наводки — то, что не сводится к
    чистым тонам.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * samplerate)
    t = np.arange(n) / samplerate

    base_freq = 90.0  # Гц — основная частота гула
    hum = np.zeros(n)
    for k, amp in enumerate([1.0, 0.6, 0.35, 0.2], start=1):
        hum += amp * np.sin(2 * np.pi * k * base_freq * t)

    white = rng.standard_normal(n)

    noise = 0.6 * _normalize_rms(hum) + 0.4 * _normalize_rms(white)
    return _normalize_rms(noise)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Смешивает речь и шум так, чтобы у смеси было ровно заданное ОСШ snr_db (дБ).

    Мощность сигнала — среднее значение его квадрата: P = mean(x**2). ОСШ по
    определению — это P_речь / P_шум, выраженное в децибелах:

        ОСШ(дБ) = 10 * log10(P_речь / P_шум).

    Готового шума с нужной мощностью у нас нет — есть некоторый noise с мощностью
    P_шум, и его нужно умножить на коэффициент k так, чтобы получить целевое ОСШ.
    Так как мощность растёт с КВАДРАТОМ амплитуды, после умножения на k мощность
    шума станет k**2 * P_шум. Подставляем это в определение ОСШ и решаем относительно k:

        snr_db = 10 * log10(P_речь / (k**2 * P_шум))
        10**(snr_db/10) = P_речь / (k**2 * P_шум)
        k**2 = P_речь / (P_шум * 10**(snr_db/10))
        k = sqrt(P_речь / (P_шум * 10**(snr_db/10)))

    Именно эта формула — ниже. Возвращает (смесь, речь, шум) — согласованно
    отмасштабированные, если потребовалась защита от клиппинга.
    """
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
    """Режет сигнал на короткие перекрывающиеся окна (кадры) по frame_len отсчётов
    со сдвигом hop_len между соседними кадрами. Это и есть операция "framing" —
    основа всех кратковременных оценок ниже (ОСШ, энергия для VAD): вместо одного
    числа на весь сигнал считаем его отдельно в каждом маленьком окне, чтобы видеть,
    как характеристика сигнала меняется во времени.
    """
    n_frames = max(1, 1 + (len(signal) - frame_len) // hop_len)
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = signal[start : start + frame_len]
    return frames


def estimate_windowed_snr(
    speech: np.ndarray, noise: np.ndarray, samplerate: int = SAMPLERATE, win_ms: float = 32.0, hop_ms: float = 16.0
) -> tuple[np.ndarray, np.ndarray]:
    """Кратковременное (покадровое) ОСШ — то же самое определение ОСШ, что и в
    mix() (10 * log10(P_речь / P_шум)), но посчитанное отдельно в каждом коротком
    окне длиной win_ms, а не для всей записи разом.

    Одно число ОСШ на всю запись слишком грубое: пока говорят — одно соотношение
    мощностей, в паузе — совсем другое (речи там нет вообще, остаётся только шум,
    и локальное ОСШ проваливается вниз). Разбиение на кадры (framing) позволяет
    увидеть эту динамику и служит основой для VAD ниже. Возвращает (время центров
    окон, ОСШ)."""
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
    """Энергетический детектор речевой активности (Voice Activity Detection).

    Идея простая: пока говорят, энергия кадра заметно выше, чем в паузе, где
    остаётся только фоновый шум. Порог строится в два шага:

        1. noise_floor_db — floor_percentile-й процентиль энергии по всей записи,
           то есть уровень, ниже которого лежат самые тихие floor_percentile%
           кадров. Если пауз в записи заметно больше, чем активной речи, это
           разумная оценка "типичного уровня тишины", не требующая знать шум заранее.
        2. threshold_db = noise_floor_db + threshold_offset_db — к полу добавлен
           запас в decibel'ах, чтобы обычные колебания шума не принимались за речь.

    Кадр помечается как "speech", если его энергия (в дБ) выше threshold_db, и как
    "silence" иначе. Соседние кадры с одинаковой меткой объединяются в один
    сегмент. Возвращает таймлайн [{start, end, label}]."""
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
