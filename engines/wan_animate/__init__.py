# Modified for Local FaceSwap macOS: Apple Silicon (MPS) support and pipeline changes.
# Wan2.2-Animate-14B (Apache-2.0, Alibaba Wan team), ported to Apple Silicon (MPS).
# Upstream: https://github.com/Wan-Video/Wan2.2 (commit 1ea34ff)
from .animate import WanAnimate

__all__ = ['WanAnimate']
