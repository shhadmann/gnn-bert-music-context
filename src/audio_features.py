"""
Audio feature extraction for the GNN-BERT music context project.
Implements: resampling, log-mel / chroma extraction, per-track normalization,
and fixed-window segmentation with per-segment feature pooling (used later
by graph_builder.py to build node features).
"""

import numpy as np
import librosa
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_audio(path, sr=22050):
    """Load an audio file, resampled to the target sample rate."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extract_log_mel(y, sr, n_mels=128):
    """Full-track log-mel spectrogram. Shape: (n_mels, time_frames)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def extract_chroma(y, sr, n_chroma=12):
    """Full-track chroma features. Shape: (n_chroma, time_frames)."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    return chroma


def normalize_per_track(features):
    """Z-score normalization using this track's own mean/std."""
    mean = features.mean()
    std = features.std()
    if std < 1e-8:
        std = 1e-8  # avoid division by zero on silent/constant input
    return (features - mean) / std


def segment_audio(y, sr, segment_duration_sec=8):
    """Split raw audio into fixed-length segments (samples, not features).
    Returns a list of 1D numpy arrays, one per segment.
    Drops a final segment shorter than half the target duration.
    """
    segment_len = int(segment_duration_sec * sr)
    segments = []
    for start in range(0, len(y), segment_len):
        seg = y[start:start + segment_len]
        if len(seg) >= segment_len // 2:  # keep segments at least half-length
            segments.append(seg)
    return segments


def extract_segment_features(y, sr, segment_duration_sec=8,
                               feature_type="chroma", n_mels=128, n_chroma=12):
    """Split audio into segments and extract one pooled feature vector per
    segment (mean over time within the segment). These become node features
    for graph_builder.py's segment graphs.

    feature_type: "chroma" or "mel"
    Returns: list of 1D numpy arrays, one per segment.
    """
    segments = segment_audio(y, sr, segment_duration_sec)
    features = []
    for seg in segments:
        if feature_type == "chroma":
            feat = extract_chroma(seg, sr, n_chroma)
        elif feature_type == "mel":
            feat = extract_log_mel(seg, sr, n_mels)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        feat = normalize_per_track(feat)
        pooled = feat.mean(axis=1)  # mean over time -> 1D vector per segment
        features.append(pooled)
    return features


def process_track(path, config):
    """End-to-end: load a track and extract everything needed downstream.
    Returns a dict with the full-track log-mel (for the CNN baseline) and
    segment-level features (for graph construction).
    """
    audio_cfg = config["audio"]
    y, sr = load_audio(path, sr=audio_cfg["sample_rate"])

    full_log_mel = normalize_per_track(
        extract_log_mel(y, sr, n_mels=audio_cfg["n_mels"])
    )

    segment_feats = extract_segment_features(
        y, sr,
        segment_duration_sec=audio_cfg["segment_duration_sec"],
        feature_type="chroma",
        n_mels=audio_cfg["n_mels"],
        n_chroma=audio_cfg["n_chroma"],
    )

    return {
        "full_log_mel": full_log_mel,       # shape: (n_mels, time)
        "segment_features": segment_feats,  # list of (n_chroma,) vectors
        "duration_sec": len(y) / sr,
        "n_segments": len(segment_feats),
    }