import sys
sys.path.append("src")

from audio_features import load_config, process_track

config = load_config("config.yaml")
result = process_track("data/raw/gtzan/genres_original/blues/blues.00000.wav", config)

print("Duration (sec):", result["duration_sec"])
print("Full log-mel shape:", result["full_log_mel"].shape)
print("Number of segments:", result["n_segments"])
print("First segment feature vector shape:", result["segment_features"][0].shape)
print("First segment feature vector (first 5 values):", result["segment_features"][0][:5])