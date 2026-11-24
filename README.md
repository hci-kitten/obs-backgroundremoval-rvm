# obs-backgroundremoval-rvm

An independent, Linux-focused fork of [`royshil/obs-backgroundremoval`](https://github.com/royshil/obs-backgroundremoval) (`v1.4.1`, `GPL-3.0-or-later`) focused on real-time 60 fps **Robust Video Matting (RVM) ResNet50** portrait matting in OBS Studio.

> **Why this fork exists:** Upstream `royshil/obs-backgroundremoval` `v1.5+` transitioned to a shader-only MediaPipe pipeline and removed ONNX Runtime, OpenCV, and RVM model support. This fork builds on `v1.4.1` to keep full-precision (`fp32`) ONNX Runtime RVM ResNet50 matting alive on Linux with asynchronous inference and zero-copy NVIDIA CUDA–OpenGL acceleration.

> **Platform support (v1):** Linux only (x86_64 and arm64). Windows and macOS are unsupported in v1.

## Highlights

- **RVM ResNet50 HQ (720p, full `fp32` precision):** Runs `rvm_resnet50_fp32.onnx` at `1280×768` (`downsample_ratio = 0.25`) with properly sized ResNet50 recurrent states (`16/32/64/128`).
- **On-demand SHA-256 verified model download:** `rvm_resnet50_fp32.onnx` (`107,479,165` bytes, `GPL-3.0`) is not bundled in git. When selected for the first time, it downloads asynchronously in the background without blocking OBS's UI or render thread, verifies SHA-256 (`25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c`), and caches under OBS's writable plugin config directory (`models/rvm_resnet50_fp32.onnx`). Set `OBS_BGREMOVAL_RVM_HQ_URL` (must be `https://`) to use a custom mirror.
- **Non-blocking latest-frame worker thread:** Moves ONNX Runtime inference off OBS's 60 Hz tick/render thread onto a dedicated 3-slot latest-frame-wins worker while preserving strict sequential RVM recurrent state updates.
- **Zero-copy CUDA–OpenGL + `fp32` `Ort::IoBinding`:** On Linux systems with an NVIDIA driver (`libcuda.so.1`, loaded dynamically via `dlopen`) and CUDA-enabled `libonnxruntime`, camera frames, RVM recurrent states (`r1..r4`), and alpha masks stay resident on the GPU via OpenGL interop and PTX pre/post-processing.
- **Graceful fallback everywhere:** Automatically degrades cleanly to CPU execution (or pre-downscaled staging + `Ort::IoBinding`) when no NVIDIA GPU, `libcuda.so.1`, CUDA Execution Provider, or OpenGL interop is available. ORT-format (`.ort`) models stay on CPU automatically.

## Recommended Settings for RVM HQ

1. Select **Robust Video Matting HQ (ResNet50, 720p)** under **Segmentation Model**.
2. Select **GPU (CUDA)** under **Inference Device** if you have an NVIDIA GPU and CUDA-enabled ONNX Runtime (otherwise **CPU** works out-of-the-box).
3. **Disable `Enable Threshold`** and set **`Temporal Smooth Factor` to `0.0`** — RVM HQ produces clean soft alpha mattes with built-in recurrent temporal stability; extra thresholding or EMA smoothing adds unnecessary latency and edge artifacts.

## Building from Source (Linux)

```bash
git clone https://github.com/hci-kitten/obs-backgroundremoval-rvm.git
cd obs-backgroundremoval-rvm
cmake --preset linux
cmake --build --preset linux
```

- Requires OBS Studio headers (`libobs-dev`), `libcurl`, `OpenCV`, and `ONNX Runtime` (`>= 1.17`, `1.29` tested with CUDA EP).
- `ENABLE_CUDA_EP` defaults to `ON` on Linux and uses the ONNX Runtime C++ `OrtApi` table with runtime CPU fallback, so CPU-only `libonnxruntime-dev` builds (such as Debian CI) compile and run without extra CUDA build dependencies.

## Credits and Licenses

- **Plugin (`GPL-3.0-or-later`):** Forked from [`royshil/obs-backgroundremoval`](https://github.com/royshil/obs-backgroundremoval) `v1.4.1` by **Roy Shilkrot**, **Kaito Udagawa**, and the `obs-backgroundremoval` contributors. See [`LICENSE`](LICENSE).
- **RVM ResNet50 model (`GPL-3.0`):** `rvm_resnet50_fp32.onnx` is created and published by **Peter Lin et al.** in [`PeterL1n/RobustVideoMatting`](https://github.com/PeterL1n/RobustVideoMatting) (`v1.0.0`, `GPL-3.0`).
- Other bundled `.ort` model weights retain their original attributions and `.license` files under `data/models/`.

---

> SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>  
> SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>  
> SPDX-FileCopyrightText: 2026 hci-kitten  
>
> SPDX-License-Identifier: GPL-3.0-or-later  
