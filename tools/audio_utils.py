import io
import os
import subprocess
import wave

import av
import numpy as np
import torch


def _audio_layout_name(channels: int) -> str:
    return "mono" if channels == 1 else "stereo"


def _decode_audio_array(path: str, sample_rate: int | None = None, channels: int | None = None) -> tuple[np.ndarray, int]:
    with av.open(path) as container:
        stream = container.streams.audio[0]
        source_rate = int(stream.codec_context.sample_rate or stream.rate or sample_rate or 16000)
        if channels is None:
            channels = int(stream.codec_context.channels or 1)
        resampler = av.AudioResampler(format="fltp", layout=_audio_layout_name(channels), rate=sample_rate or source_rate)
        chunks = []

        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                chunks.append(out_frame.to_ndarray())

        for out_frame in resampler.resample(None):
            chunks.append(out_frame.to_ndarray())

    if not chunks:
        raise RuntimeError(f"Failed to decode audio: {path}")

    audio = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(audio), int(sample_rate or source_rate)


def load_audio_tensor(path: str):
    audio, sample_rate = _decode_audio_array(path)
    return torch.from_numpy(audio.copy()).float(), sample_rate


def load_audio_mono(path: str, sample_rate: int) -> np.ndarray:
    audio, _ = _decode_audio_array(path, sample_rate=sample_rate, channels=1)
    return audio[0]


def _resample_audio_array(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    channels = 1 if audio.ndim == 1 else int(audio.shape[0])
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout=_audio_layout_name(channels))
    frame.sample_rate = int(src_sr)
    resampler = av.AudioResampler(format="fltp", layout=_audio_layout_name(channels), rate=int(dst_sr))

    chunks = []
    for out_frame in resampler.resample(frame):
        chunks.append(out_frame.to_ndarray())
    for out_frame in resampler.resample(None):
        chunks.append(out_frame.to_ndarray())

    if not chunks:
        raise RuntimeError(f"Failed to resample audio from {src_sr} to {dst_sr}")
    return np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float32)


def resample_audio_tensor(audio_tensor: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return audio_tensor
    original_device = audio_tensor.device
    original_dtype = audio_tensor.dtype
    if audio_tensor.dim() == 1:
        audio_np = audio_tensor.detach().cpu().float().numpy()[np.newaxis, :]
        squeeze = True
    else:
        audio_np = audio_tensor.detach().cpu().float().numpy()
        squeeze = False
    result = torch.from_numpy(_resample_audio_array(audio_np, src_sr, dst_sr).copy()).to(dtype=original_dtype)
    result = result.squeeze(0) if squeeze else result
    return result.to(original_device)


def change_speed_int16(input_audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    raw_audio = input_audio.astype(np.int16, copy=False).tobytes()
    process = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-filter:a",
            f"atempo={speed}",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        input=raw_audio,
        check=True,
        capture_output=True,
    )
    return np.frombuffer(process.stdout, dtype=np.int16)


def _normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"Only mono audio is supported, got shape {array.shape}")
    return np.ascontiguousarray(array)


def _audio_frame_from_mono(audio: np.ndarray) -> av.AudioFrame:
    if np.issubdtype(audio.dtype, np.floating):
        pcm = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)[np.newaxis, :]
        return av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm), format="fltp", layout="mono")
    if audio.dtype == np.int16:
        return av.AudioFrame.from_ndarray(audio[np.newaxis, :], format="s16", layout="mono")
    if audio.dtype == np.int32:
        return av.AudioFrame.from_ndarray(audio[np.newaxis, :], format="s32", layout="mono")
    raise ValueError(f"Unsupported audio dtype for encoding: {audio.dtype}")


def write_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = _normalize_audio_array(audio)
    if np.issubdtype(audio.dtype, np.floating):
        pcm = np.clip(audio, -1.0, 1.0)
        sample_width = 2
        raw = (pcm * 32767.0).astype(np.int16).tobytes()
    elif audio.dtype == np.int16:
        sample_width = 2
        raw = audio.tobytes()
    elif audio.dtype == np.int32:
        sample_width = 4
        raw = audio.tobytes()
    else:
        raise ValueError(f"Unsupported audio dtype for WAV: {audio.dtype}")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(raw)
    return buffer.getvalue()


def write_wav_file(path: str, audio: np.ndarray, sample_rate: int) -> None:
    with open(path, "wb") as fw:
        fw.write(write_wav_bytes(audio, sample_rate))


def write_ogg_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = _normalize_audio_array(audio)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="ogg") as container:
        stream = container.add_stream("libvorbis", rate=int(sample_rate))
        stream.layout = "mono"
        frame = _audio_frame_from_mono(audio)
        frame.sample_rate = int(sample_rate)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def write_audio_file(path: str, audio: np.ndarray, sample_rate: int, format: str | None = None) -> None:
    target_format = (format or os.path.splitext(path)[1].lstrip(".") or "wav").lower()
    if target_format == "wav":
        write_wav_file(path, audio, sample_rate)
        return
    if target_format == "ogg":
        payload = write_ogg_bytes(audio, sample_rate)
        with open(path, "wb") as fw:
            fw.write(payload)
        return
    raise ValueError(f"Unsupported output audio format: {target_format}")
