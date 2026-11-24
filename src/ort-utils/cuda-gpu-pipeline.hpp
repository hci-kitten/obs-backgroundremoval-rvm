// SPDX-FileCopyrightText: 2026 hci-kitten
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CUDA_GPU_PIPELINE_HPP
#define CUDA_GPU_PIPELINE_HPP

#include <onnxruntime_cxx_api.h>
#include <obs-module.h>
#include <graphics/graphics.h>

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

class CudaGpuPipeline {
public:
	static constexpr int kInputSlotCount = 3;

	enum class GlInteropStatus {
		Ok,
		FallbackReleased,
		FallbackOwnershipPinned,
	};

	CudaGpuPipeline();
	~CudaGpuPipeline();

	CudaGpuPipeline(const CudaGpuPipeline &) = delete;
	CudaGpuPipeline &operator=(const CudaGpuPipeline &) = delete;

	// Initialize device-0 primary CUDA context, streams, PTX kernels, and ORT CUDA buffers.
	// Must be called under modelMutex after session shapes are populated.
	bool initialize(uint32_t width, uint32_t height, bool hasRecurrentStates, float downsampleRatio,
			const std::vector<std::vector<int64_t>> &inputDims,
			const std::vector<std::vector<int64_t>> &outputDims);

	// Unregister OpenGL resources while OBS graphics context is active.
	// Returns true only when CUDA confirmed unmap (if mapped) + unregister; false if any handle remains pinned.
	bool releaseGlResourcesOnGraphicsThread();
	bool releaseGlInputResourceOnGraphicsThread();
	bool releaseGlAlphaResourceOnGraphicsThread();

	// Release CUDA tensor allocations, worker stream, and PTX module (retains primary context if any GL resource remains pinned).
	void shutdown();

	bool isInitialized() const { return initialized_.load(std::memory_order_acquire); }
	bool hasKernels() const { return kernelsReady_.load(std::memory_order_acquire); }
	bool isGlInteropDisabled() const { return glInteropDisabled_.load(std::memory_order_acquire); }
	uint32_t width() const { return width_; }
	uint32_t height() const { return height_; }

	// Graphics-thread zero-copy capture: maps OpenGL RGBA8 texture and copies D2D into slotIdx.
	// FallbackOwnershipPinned means unmap/unregister failed and caller MUST quarantine gsTex without GL reuse/destroy.
	GlInteropStatus captureGlTextureToSlotOnGraphicsThread(gs_texture_t *gsTex, int slotIdx);

	// Graphics-thread zero-copy mask upload: copies latest GPU R8 mask D2D into persistent OpenGL GS_R8 texture.
	// FallbackOwnershipPinned means unmap/unregister failed and caller MUST quarantine gsAlphaTex without GL destroy.
	GlInteropStatus uploadMaskToGlTextureOnGraphicsThread(gs_texture_t *gsAlphaTex);

	// Fallback H2D raw RGBA/BGRA upload from downscaled stage surface into slotIdx (worker or render thread).
	bool uploadHostFrameToSlot(const uint8_t *hostData, uint32_t hostLinesize, int slotIdx, bool isBgra);

	// Zero RVM recurrent states r1..r4 and previous temporal mask in VRAM (on activation or model reset).
	void resetRecurrentAndTemporalState();

	// Run full GPU-resident inference on slotIdx:
	//   1) PTX RGBA/BGRA -> NCHW RGB fp32 [0,1] into CUDA d_src_
	//   2) Ort::IoBinding Run with GPU-resident r1..r4 ping-pong + d_downsampleRatio_ + d_pha_
	//   3) PTX pha fp32 -> inverted/thresholded (+ optional EMA smoothed) R8 mask in d_maskScratch_
	// If copyMaskToHost != nullptr, also copies the 1-channel R8 mask (width * height bytes) to host buffer
	// (used when OpenCV contour morphology is enabled or when GL mask interop fell back to gs_texture_set_image).
	bool runInferenceOnSlot(Ort::Session &session, const std::vector<Ort::AllocatedStringPtr> &inputNames,
				const std::vector<Ort::AllocatedStringPtr> &outputNames, int slotIdx,
				bool enableThreshold, float threshold, float temporalSmoothFactor,
				uint8_t *copyMaskToHost);

	// Copy finished dMaskScratch -> dMaskPublished (call under outputLock before bumping maskGeneration).
	bool publishScratchMask();

	// Fallback readback of dMaskPublished -> hostDst (width * height bytes) if GL R8 interop declines on render thread.
	bool downloadPublishedMaskToHost(uint8_t *hostDst);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
	uint32_t width_ = 0;
	uint32_t height_ = 0;
	std::atomic<bool> initialized_{false};
	std::atomic<bool> kernelsReady_{false};
	std::atomic<bool> glInteropDisabled_{false};
};

#endif /* CUDA_GPU_PIPELINE_HPP */
