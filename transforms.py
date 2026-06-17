# T_safe: safe acquisition-level transforms for the Waveform Consistency
# Objective (WCO). These simulate recording-nuisance variation (gain, baseline,
# noise, alignment) WITHOUT altering waveform morphology. See proposal 4.4.
#
# NOTE (see RESEARCH_CRITIQUE.md): these model only minor nuisance shift, not
# the dominant montage/site shift (Fpz-Cz vs C4-A1). Kept faithful to the
# proposal; revisit when extending to SHHS.
import torch


def amplitude_scale(x, lo=0.8, hi=1.2):
    b = x.size(0)
    s = torch.empty(b, 1, 1, device=x.device).uniform_(lo, hi)
    return x * s


def dc_offset(x, max_uv=5.0):
    # signals are z-scored per recording, so 5 uV is approximate; use a small
    # fraction of the per-sample std as a stand-in offset.
    b = x.size(0)
    off = torch.empty(b, 1, 1, device=x.device).uniform_(-max_uv, max_uv)
    off = off * 0.01 * x.std()
    return x + off


def gaussian_noise(x, min_snr_db=20.0):
    sig_power = x.pow(2).mean(dim=(1, 2), keepdim=True)
    snr = 10 ** (min_snr_db / 10.0)
    noise_power = sig_power / snr
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise


def time_shift(x, max_frac=0.5, fs=100):
    max_shift = int(max_frac * fs)
    if max_shift == 0:
        return x
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(x, shifts=shift, dims=-1)


def apply_safe_transforms(x, fs=100):
    """Compose all four safe transforms with fresh randomness."""
    x = amplitude_scale(x)
    x = dc_offset(x)
    x = gaussian_noise(x)
    x = time_shift(x, fs=fs)
    return x
