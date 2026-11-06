# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for Local FaceSwap macOS: Apple Silicon (MPS) support and pipeline changes.
# MPS port: the full ProcessPipeline (SAM2, FLUX-Kontext, moviepy) is not
# imported eagerly; headswap preprocessing only needs pose2d + utils.
