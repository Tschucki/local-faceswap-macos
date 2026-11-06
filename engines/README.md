# engines

Model code bundled with Local FaceSwap macOS. Every part comes from an Apache-2.0
project and keeps its copyright headers. Files changed for this project
(Apple Silicon / MPS support, in-memory inputs, LoRA merging, pipeline hooks)
say so in their header.

| Folder | What it is | Origin |
|---|---|---|
| `wan_animate/` | Wan2.2-Animate-14B, replacement mode (head + hair) | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) |
| `dreamidv_wan/`, `dreamidv_prep.py` | DreamID-V face model on Wan2.1-1.3B (face pass, face-only mode) | [bytedance/DreamID-V](https://github.com/bytedance/DreamID-V) |
| `express_adaption/` | Face landmarks and pose alignment for DreamID-V (contains MediaPipe code) | DreamID-V, [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) |

`app.py`, `headswap.py`, `faceswap.py` and `compare.py` add this folder to `sys.path`,
so the packages are imported under their own names.
