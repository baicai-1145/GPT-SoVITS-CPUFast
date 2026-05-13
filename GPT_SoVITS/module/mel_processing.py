import torch
import torch.utils.data

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / torch.log(torch.tensor(6.4, dtype=freq.dtype, device=freq.device))
    linear = 3.0 * freq / 200.0
    log = min_log_mel + torch.log(freq / min_log_hz) * logstep
    return torch.where(freq >= min_log_hz, log, linear)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    min_log_hz = 1000.0
    min_log_mel = 15.0
    logstep = torch.log(torch.tensor(6.4, dtype=mels.dtype, device=mels.device)) / 27.0
    linear = 200.0 * mels / 3.0
    log = min_log_hz * torch.exp(logstep * (mels - min_log_mel))
    return torch.where(mels >= min_log_mel, log, linear)


def mel_filter_bank(sr, n_fft, n_mels, fmin, fmax, dtype, device):
    if fmax is None:
        fmax = float(sr) / 2.0
    fft_freqs = torch.linspace(0, float(sr) / 2.0, int(1 + n_fft // 2), dtype=dtype, device=device)
    mel_min = _hz_to_mel(torch.tensor(float(fmin), dtype=dtype, device=device))
    mel_max = _hz_to_mel(torch.tensor(float(fmax), dtype=dtype, device=device))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype, device=device)
    hz_points = _mel_to_hz(mel_points)

    ramps = hz_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / (hz_points[1:-1] - hz_points[:-2])[:, None]
    upper = ramps[2:] / (hz_points[2:] - hz_points[1:-1])[:, None]
    weights = torch.maximum(torch.zeros((), dtype=dtype, device=device), torch.minimum(lower, upper))
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    return weights * enorm[:, None]


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.2:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.2:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    # wnsize_dtype_device = str(win_size) + '_' + dtype_device
    key = "%s-%s-%s-%s-%s" % (dtype_device, n_fft, sampling_rate, hop_size, win_size)
    # if wnsize_dtype_device not in hann_window:
    if key not in hann_window:
        # hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
        hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)
    # spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    # fmax_dtype_device = str(fmax) + '_' + dtype_device
    key = "%s-%s-%s-%s-%s-%s" % (dtype_device, n_fft, num_mels, sampling_rate, fmin, fmax)
    # if fmax_dtype_device not in mel_basis:
    if key not in mel_basis:
        mel_basis[key] = mel_filter_bank(sampling_rate, n_fft, num_mels, fmin, fmax, spec.dtype, spec.device)
    # spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.2:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.2:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    # fmax_dtype_device = str(fmax) + '_' + dtype_device
    fmax_dtype_device = "%s-%s-%s-%s-%s-%s-%s-%s" % (
        dtype_device,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
    )
    # wnsize_dtype_device = str(win_size) + '_' + dtype_device
    wnsize_dtype_device = fmax_dtype_device
    if fmax_dtype_device not in mel_basis:
        mel_basis[fmax_dtype_device] = mel_filter_bank(
            sampling_rate, n_fft, num_mels, fmin, fmax, y.dtype, y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
