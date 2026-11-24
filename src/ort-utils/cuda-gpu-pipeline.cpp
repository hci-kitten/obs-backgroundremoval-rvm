// SPDX-FileCopyrightText: 2026 hci-kitten
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "cuda-gpu-pipeline.hpp"

#include "../plugin-support.h"

#include <algorithm>
#include <array>
#include <cstring>

#if defined(__linux__) && defined(HAVE_ONNXRUNTIME_CUDA_EP)
#include <dlfcn.h>

namespace {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUcontext = struct CUctx_st *;
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using CUstream = struct CUstream_st *;
using CUarray = struct CUarray_st *;
using CUgraphicsResource = struct CUgraphicsResource_st *;
using GLuint = unsigned int;
using GLenum = unsigned int;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr GLenum kGlTexture2D = 0x0DE1;
constexpr unsigned int kCuStreamNonBlocking = 0x1;
constexpr unsigned int kCuGraphicsRegisterFlagsReadOnly = 0x1;
constexpr unsigned int kCuGraphicsRegisterFlagsWriteDiscard = 0x2;
constexpr unsigned int kCuGraphicsMapResourceFlagsReadOnly = 0x1;
constexpr unsigned int kCuGraphicsMapResourceFlagsWriteDiscard = 0x2;

enum CUmemorytype {
	CU_MEMORYTYPE_HOST = 0x01,
	CU_MEMORYTYPE_DEVICE = 0x02,
	CU_MEMORYTYPE_ARRAY = 0x03,
	CU_MEMORYTYPE_UNIFIED = 0x04
};

enum CUjit_option {
	CU_JIT_ERROR_LOG_BUFFER = 5,
	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
};

struct CUDA_MEMCPY2D_v2 {
	size_t srcXInBytes;
	size_t srcY;
	CUmemorytype srcMemoryType;
	const void *srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	size_t srcPitch;

	size_t dstXInBytes;
	size_t dstY;
	CUmemorytype dstMemoryType;
	void *dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	size_t dstPitch;

	size_t WidthInBytes;
	size_t Height;
};

struct CudaDriverApi {
	void *lib = nullptr;

	CUresult (*cuInit)(unsigned int) = nullptr;
	CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
	CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
	CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
	CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
	CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;

	CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = nullptr;
	CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
	CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream) = nullptr;
	CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream) = nullptr;
	CUresult (*cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream) = nullptr;
	CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream) = nullptr;
	CUresult (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D_v2 *, CUstream) = nullptr;

	CUresult (*cuStreamCreate)(CUstream *, unsigned int) = nullptr;
	CUresult (*cuStreamDestroy)(CUstream) = nullptr;
	CUresult (*cuStreamSynchronize)(CUstream) = nullptr;

	CUresult (*cuModuleLoadDataEx)(CUmodule *, const void *, unsigned int, CUjit_option *, void **) = nullptr;
	CUresult (*cuModuleUnload)(CUmodule) = nullptr;
	CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
	CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
				   unsigned int, unsigned int, CUstream, void **, void **) = nullptr;

	CUresult (*cuGraphicsGLRegisterImage)(CUgraphicsResource *, GLuint, GLenum, unsigned int) = nullptr;
	CUresult (*cuGraphicsUnregisterResource)(CUgraphicsResource) = nullptr;
	CUresult (*cuGraphicsResourceSetMapFlags)(CUgraphicsResource, unsigned int) = nullptr;
	CUresult (*cuGraphicsMapResources)(unsigned int, CUgraphicsResource *, CUstream) = nullptr;
	CUresult (*cuGraphicsUnmapResources)(unsigned int, CUgraphicsResource *, CUstream) = nullptr;
	CUresult (*cuGraphicsSubResourceGetMappedArray)(CUarray *, CUgraphicsResource, unsigned int,
							unsigned int) = nullptr;

	template<typename Fn> bool loadSym(Fn &out, const char *primary, const char *fallback = nullptr)
	{
		void *sym = dlsym(lib, primary);
		if (!sym && fallback) {
			sym = dlsym(lib, fallback);
		}
		if (!sym) {
			return false;
		}
		std::memcpy(&out, &sym, sizeof(Fn));
		return true;
	}

	bool load()
	{
		if (lib) {
			return true;
		}
		lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
		if (!lib) {
			obs_log(LOG_WARNING, "dlopen(libcuda.so.1) failed: %s", dlerror());
			return false;
		}
		const bool ok =
			loadSym(cuInit, "cuInit") && loadSym(cuDeviceGet, "cuDeviceGet") &&
			loadSym(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
			loadSym(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2",
				"cuDevicePrimaryCtxRelease") &&
			loadSym(cuCtxSetCurrent, "cuCtxSetCurrent") &&
			loadSym(cuGetErrorString, "cuGetErrorString") &&
			loadSym(cuMemAlloc, "cuMemAlloc_v2", "cuMemAlloc") &&
			loadSym(cuMemFree, "cuMemFree_v2", "cuMemFree") &&
			loadSym(cuMemsetD8Async, "cuMemsetD8Async") &&
			loadSym(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync") &&
			loadSym(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync") &&
			loadSym(cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync") &&
			loadSym(cuMemcpy2DAsync, "cuMemcpy2DAsync_v2", "cuMemcpy2DAsync") &&
			loadSym(cuStreamCreate, "cuStreamCreate") &&
			loadSym(cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy") &&
			loadSym(cuStreamSynchronize, "cuStreamSynchronize") &&
			loadSym(cuModuleLoadDataEx, "cuModuleLoadDataEx") &&
			loadSym(cuModuleUnload, "cuModuleUnload") &&
			loadSym(cuModuleGetFunction, "cuModuleGetFunction") &&
			loadSym(cuLaunchKernel, "cuLaunchKernel");
		if (!ok) {
			obs_log(LOG_WARNING, "Required CUDA Driver API symbols missing in libcuda.so.1");
			dlclose(lib);
			lib = nullptr;
			return false;
		}
		// GL interop symbols are optional; if absent, pipeline cleanly uses downscaled staging + IoBinding.
		const bool glOk =
			loadSym(cuGraphicsGLRegisterImage, "cuGraphicsGLRegisterImage") &&
			loadSym(cuGraphicsUnregisterResource, "cuGraphicsUnregisterResource") &&
			loadSym(cuGraphicsResourceSetMapFlags, "cuGraphicsResourceSetMapFlags") &&
			loadSym(cuGraphicsMapResources, "cuGraphicsMapResources") &&
			loadSym(cuGraphicsUnmapResources, "cuGraphicsUnmapResources") &&
			loadSym(cuGraphicsSubResourceGetMappedArray, "cuGraphicsSubResourceGetMappedArray");
		if (!glOk) {
			cuGraphicsGLRegisterImage = nullptr;
		}
		return true;
	}

	const char *errStr(CUresult err) const
	{
		const char *msg = "unknown";
		if (cuGetErrorString) {
			cuGetErrorString(err, &msg);
		}
		return msg ? msg : "unknown";
	}
};

// PTX 7.0 (sm_50 virtual target -> JIT-compiled natively to RTX 3090 sm_86 SASS by NVIDIA driver):
// 1) rgba8_to_nchw_rgb_f32: converts packed 4-byte RGBA8 (or BGRA8 when swap_rb != 0) into planar
//    [1, 3, H, W] float32 in [0.0, 1.0].
// 2) pha_f32_to_inv_mask_u8: converts RVM pha [1, 1, H, W] float32 in [0.0, 1.0] into OBS backgroundMask
//    uint8 (0 = foreground, 255 = background), with optional binary threshold and temporal EMA blend.
constexpr const char *kPrePostPtx = R"PTX(
.version 7.0
.target sm_50
.address_size 64

.visible .entry rgba8_to_nchw_rgb_f32(
	.param .u64 p_src,
	.param .u64 p_dst,
	.param .u32 p_w,
	.param .u32 p_h,
	.param .u32 p_swap_rb
)
{
	.reg .pred %p<3>;
	.reg .u32 %r<17>;
	.reg .u64 %rd<7>;
	.reg .f32 %f<4>;

	ld.param.u32 %r0, [p_w];
	ld.param.u32 %r1, [p_h];

	mov.u32 %r2, %ctaid.x;
	mov.u32 %r3, %ntid.x;
	mov.u32 %r4, %tid.x;
	mad.lo.u32 %r5, %r2, %r3, %r4;

	mov.u32 %r2, %ctaid.y;
	mov.u32 %r3, %ntid.y;
	mov.u32 %r4, %tid.y;
	mad.lo.u32 %r6, %r2, %r3, %r4;

	setp.ge.u32 %p0, %r5, %r0;
	setp.ge.u32 %p1, %r6, %r1;
	or.pred %p0, %p0, %p1;
	@%p0 bra DONE_PRE;

	mad.lo.u32 %r7, %r6, %r0, %r5;
	mul.wide.u32 %rd2, %r7, 4;
	ld.param.u64 %rd0, [p_src];
	add.u64 %rd3, %rd0, %rd2;
	ld.global.u32 %r8, [%rd3];

	and.b32 %r9, %r8, 255;
	shr.u32 %r10, %r8, 8;
	and.b32 %r10, %r10, 255;
	shr.u32 %r11, %r8, 16;
	and.b32 %r11, %r11, 255;

	ld.param.u32 %r12, [p_swap_rb];
	setp.ne.u32 %p2, %r12, 0;
	selp.u32 %r13, %r11, %r9, %p2;
	selp.u32 %r14, %r9, %r11, %p2;

	cvt.rn.f32.u32 %f1, %r13;
	cvt.rn.f32.u32 %f2, %r10;
	cvt.rn.f32.u32 %f3, %r14;
	div.rn.f32 %f1, %f1, 0f437F0000;
	div.rn.f32 %f2, %f2, 0f437F0000;
	div.rn.f32 %f3, %f3, 0f437F0000;

	ld.param.u64 %rd1, [p_dst];
	mul.lo.u32 %r15, %r0, %r1;

	add.u64 %rd4, %rd1, %rd2;
	st.global.f32 [%rd4], %f1;

	add.u32 %r16, %r7, %r15;
	mul.wide.u32 %rd5, %r16, 4;
	add.u64 %rd4, %rd1, %rd5;
	st.global.f32 [%rd4], %f2;

	add.u32 %r16, %r16, %r15;
	mul.wide.u32 %rd5, %r16, 4;
	add.u64 %rd4, %rd1, %rd5;
	st.global.f32 [%rd4], %f3;

DONE_PRE:
	ret;
}

.visible .entry pha_f32_to_inv_mask_u8(
	.param .u64 p_pha,
	.param .u64 p_mask,
	.param .u64 p_prev,
	.param .u32 p_n,
	.param .u32 p_enable_thresh,
	.param .u32 p_thresh_u8,
	.param .u32 p_use_smooth,
	.param .f32 p_smooth_alpha
)
{
	.reg .pred %p<4>;
	.reg .u32 %r<12>;
	.reg .u64 %rd<7>;
	.reg .f32 %f<8>;

	ld.param.u32 %r0, [p_n];
	mov.u32 %r1, %ctaid.x;
	mov.u32 %r2, %ntid.x;
	mov.u32 %r3, %tid.x;
	mad.lo.u32 %r4, %r1, %r2, %r3;

	setp.ge.u32 %p0, %r4, %r0;
	@%p0 bra DONE_POST;

	ld.param.u64 %rd0, [p_pha];
	mul.wide.u32 %rd3, %r4, 4;
	add.u64 %rd4, %rd0, %rd3;
	ld.global.f32 %f1, [%rd4];

	max.f32 %f1, %f1, 0f00000000;
	min.f32 %f1, %f1, 0f3F800000;
	mul.rn.f32 %f1, %f1, 0f437F0000;
	cvt.rni.u32.f32 %r5, %f1;

	ld.param.u32 %r6, [p_enable_thresh];
	setp.ne.u32 %p1, %r6, 0;
	@!%p1 bra SOFT_ALPHA;

	ld.param.u32 %r7, [p_thresh_u8];
	setp.lt.u32 %p2, %r5, %r7;
	selp.u32 %r8, 255, 0, %p2;
	bra AFTER_MASK;

SOFT_ALPHA:
	sub.u32 %r8, 255, %r5;

AFTER_MASK:
	cvt.u64.u32 %rd5, %r4;
	ld.param.u64 %rd2, [p_prev];
	add.u64 %rd6, %rd2, %rd5;

	ld.param.u32 %r9, [p_use_smooth];
	setp.ne.u32 %p3, %r9, 0;
	@!%p3 bra WRITE_OUT;

	ld.global.u8 %r10, [%rd6];
	ld.param.f32 %f2, [p_smooth_alpha];
	mov.f32 %f3, 0f3F800000;
	sub.rn.f32 %f4, %f3, %f2;
	cvt.rn.f32.u32 %f5, %r8;
	cvt.rn.f32.u32 %f6, %r10;
	mul.rn.f32 %f5, %f5, %f2;
	fma.rn.f32 %f7, %f6, %f4, %f5;
	max.f32 %f7, %f7, 0f00000000;
	min.f32 %f7, %f7, 0f437F0000;
	cvt.rni.u32.f32 %r8, %f7;

WRITE_OUT:
	st.global.u8 [%rd6], %r8;
	ld.param.u64 %rd1, [p_mask];
	add.u64 %rd4, %rd1, %rd5;
	st.global.u8 [%rd4], %r8;

DONE_POST:
	ret;
}
)PTX";

size_t shapeElements(const std::vector<int64_t> &dims)
{
	size_t n = 1;
	for (int64_t d : dims) {
		if (d > 0) {
			n *= static_cast<size_t>(d);
		}
	}
	return n;
}

} // namespace

struct CudaGpuPipeline::Impl {
	CudaDriverApi api;
	CUdevice device = 0;
	CUcontext primaryCtx = nullptr;
	bool primaryCtxRetained = false;

	CUstream renderStream = nullptr;
	CUstream workerStream = nullptr;
	CUmodule ptxModule = nullptr;
	CUfunction fnPreprocess = nullptr;
	CUfunction fnPostprocess = nullptr;

	// Triple-buffered RGBA/BGRA8 input slots (width * height * 4 bytes each)
	std::array<CUdeviceptr, CudaGpuPipeline::kInputSlotCount> dInputSlots{{0, 0, 0}};
	std::array<bool, CudaGpuPipeline::kInputSlotCount> slotIsBgra{{false, false, false}};

	// OpenGL registered resources (touched ONLY on OBS graphics thread)
	struct PinnedGlEntry {
		CUgraphicsResource res = nullptr;
		bool mapped = false;
	};

	CUgraphicsResource glInputRes = nullptr;
	bool glInputMapped = false;
	gs_texture_t *glInputGsTex = nullptr;
	GLuint glInputId = 0;
	uint32_t glInputW = 0;
	uint32_t glInputH = 0;
	std::vector<PinnedGlEntry> pinnedInputEntries;

	CUgraphicsResource glAlphaRes = nullptr;
	bool glAlphaMapped = false;
	gs_texture_t *glAlphaGsTex = nullptr;
	GLuint glAlphaId = 0;
	uint32_t glAlphaW = 0;
	uint32_t glAlphaH = 0;
	std::vector<PinnedGlEntry> pinnedAlphaEntries;
	bool loggedGlInteropFallback = false;

	// ORT CUDA tensors & VRAM buffers (fp32 only per Build-1 precision fence)
	Ort::MemoryInfo cudaMemInfo{"Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault};
	CUdeviceptr dSrc = 0;
	size_t srcElems = 0;
	Ort::Value ortSrc{nullptr};

	bool hasRecurrent = false;
	float downsampleRatio = 1.0f;
	int stateParity = 0;
	std::array<std::array<CUdeviceptr, 4>, 2> dStates{{{0, 0, 0, 0}, {0, 0, 0, 0}}};
	std::array<size_t, 4> stateBytes{{0, 0, 0, 0}};
	std::array<std::array<Ort::Value, 4>, 2> ortStates{
		{{Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}},
		 {Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}}}};

	CUdeviceptr dRatio = 0;
	Ort::Value ortRatio{nullptr};

	CUdeviceptr dPha = 0;
	size_t phaElems = 0;
	Ort::Value ortPha{nullptr};

	CUdeviceptr dMaskScratch = 0;
	CUdeviceptr dMaskPublished = 0;
	CUdeviceptr dPrevMask = 0;
	size_t maskBytes = 0;
	bool hasPrevMask = false;

	bool bindCurrentCtx() const
	{
		if (!primaryCtxRetained || !api.cuCtxSetCurrent) {
			return false;
		}
		return api.cuCtxSetCurrent(primaryCtx) == CUDA_SUCCESS;
	}

	bool releaseOneGlResource(CUgraphicsResource &res, bool &mapped)
	{
		if (!res) {
			mapped = false;
			return true;
		}
		if (mapped) {
			if (!api.cuGraphicsUnmapResources ||
			    api.cuGraphicsUnmapResources(1, &res, renderStream) != CUDA_SUCCESS) {
				return false;
			}
			mapped = false;
		}
		if (!api.cuGraphicsUnregisterResource || api.cuGraphicsUnregisterResource(res) != CUDA_SUCCESS) {
			return false;
		}
		res = nullptr;
		return true;
	}

	bool drainPinnedList(std::vector<PinnedGlEntry> &entries)
	{
		auto it = entries.begin();
		while (it != entries.end()) {
			if (releaseOneGlResource(it->res, it->mapped)) {
				it = entries.erase(it);
			} else {
				++it;
			}
		}
		return entries.empty();
	}

	bool hasAnyGlHandles() const
	{
		return glInputRes != nullptr || glAlphaRes != nullptr || !pinnedInputEntries.empty() ||
		       !pinnedAlphaEntries.empty();
	}
};

CudaGpuPipeline::CudaGpuPipeline() : impl_(std::make_unique<Impl>()) {}

CudaGpuPipeline::~CudaGpuPipeline()
{
	shutdown();
}

bool CudaGpuPipeline::releaseGlInputResourceOnGraphicsThread()
{
	if (!impl_ || !impl_->primaryCtxRetained) {
		return impl_ ? (impl_->glInputRes == nullptr && impl_->pinnedInputEntries.empty()) : true;
	}
	if (!impl_->bindCurrentCtx()) {
		return !impl_->glInputRes && impl_->pinnedInputEntries.empty();
	}
	const bool activeReleased = impl_->releaseOneGlResource(impl_->glInputRes, impl_->glInputMapped);
	if (activeReleased) {
		impl_->glInputGsTex = nullptr;
		impl_->glInputId = 0;
		impl_->glInputW = 0;
		impl_->glInputH = 0;
	}
	const bool pinnedReleased = impl_->drainPinnedList(impl_->pinnedInputEntries);
	return activeReleased && pinnedReleased;
}

bool CudaGpuPipeline::releaseGlAlphaResourceOnGraphicsThread()
{
	if (!impl_ || !impl_->primaryCtxRetained) {
		return impl_ ? (impl_->glAlphaRes == nullptr && impl_->pinnedAlphaEntries.empty()) : true;
	}
	if (!impl_->bindCurrentCtx()) {
		return !impl_->glAlphaRes && impl_->pinnedAlphaEntries.empty();
	}
	const bool activeReleased = impl_->releaseOneGlResource(impl_->glAlphaRes, impl_->glAlphaMapped);
	if (activeReleased) {
		impl_->glAlphaGsTex = nullptr;
		impl_->glAlphaId = 0;
		impl_->glAlphaW = 0;
		impl_->glAlphaH = 0;
	}
	const bool pinnedReleased = impl_->drainPinnedList(impl_->pinnedAlphaEntries);
	return activeReleased && pinnedReleased;
}

bool CudaGpuPipeline::releaseGlResourcesOnGraphicsThread()
{
	const bool inOk = releaseGlInputResourceOnGraphicsThread();
	const bool alphaOk = releaseGlAlphaResourceOnGraphicsThread();
	return inOk && alphaOk;
}

void CudaGpuPipeline::shutdown()
{
	if (!impl_) {
		return;
	}
	const bool keepGlOwnerAlive = impl_->hasAnyGlHandles();
	if (impl_->primaryCtxRetained && impl_->bindCurrentCtx()) {
		if (impl_->renderStream) {
			impl_->api.cuStreamSynchronize(impl_->renderStream);
		}
		if (impl_->workerStream) {
			impl_->api.cuStreamSynchronize(impl_->workerStream);
		}

		impl_->ortSrc = Ort::Value{nullptr};
		impl_->ortRatio = Ort::Value{nullptr};
		impl_->ortPha = Ort::Value{nullptr};
		for (int p = 0; p < 2; p++) {
			for (int i = 0; i < 4; i++) {
				impl_->ortStates[p][i] = Ort::Value{nullptr};
			}
		}

		auto freeDev = [&](CUdeviceptr &ptr) {
			if (ptr != 0) {
				impl_->api.cuMemFree(ptr);
				ptr = 0;
			}
		};

		for (auto &slotPtr : impl_->dInputSlots) {
			freeDev(slotPtr);
		}
		freeDev(impl_->dSrc);
		for (int p = 0; p < 2; p++) {
			for (int i = 0; i < 4; i++) {
				freeDev(impl_->dStates[p][i]);
			}
		}
		freeDev(impl_->dRatio);
		freeDev(impl_->dPha);
		freeDev(impl_->dMaskScratch);
		freeDev(impl_->dMaskPublished);
		freeDev(impl_->dPrevMask);

		if (impl_->ptxModule) {
			impl_->api.cuModuleUnload(impl_->ptxModule);
			impl_->ptxModule = nullptr;
		}
		impl_->fnPreprocess = nullptr;
		impl_->fnPostprocess = nullptr;

		if (impl_->workerStream) {
			impl_->api.cuStreamDestroy(impl_->workerStream);
			impl_->workerStream = nullptr;
		}

		if (!keepGlOwnerAlive) {
			if (impl_->renderStream) {
				impl_->api.cuStreamDestroy(impl_->renderStream);
				impl_->renderStream = nullptr;
			}
			impl_->api.cuDevicePrimaryCtxRelease(impl_->device);
			impl_->primaryCtx = nullptr;
			impl_->primaryCtxRetained = false;
		}
	}
	if (!keepGlOwnerAlive && impl_->api.lib) {
		dlclose(impl_->api.lib);
		impl_->api = CudaDriverApi{};
	}
	width_ = 0;
	height_ = 0;
	initialized_.store(false, std::memory_order_release);
	kernelsReady_.store(false, std::memory_order_release);
	glInteropDisabled_.store(keepGlOwnerAlive, std::memory_order_release);
}

bool CudaGpuPipeline::initialize(uint32_t width, uint32_t height, bool hasRecurrentStates, float downsampleRatio,
				 const std::vector<std::vector<int64_t>> &inputDims,
				 const std::vector<std::vector<int64_t>> &outputDims)
{
	shutdown();

	if (width == 0 || height == 0 || inputDims.empty() || outputDims.empty()) {
		return false;
	}
	if (hasRecurrentStates && (inputDims.size() < 6 || outputDims.size() < 5)) {
		obs_log(LOG_WARNING, "Unexpected RVM tensor counts (in=%zu out=%zu)", inputDims.size(),
			outputDims.size());
		return false;
	}

	if (!impl_->api.load()) {
		return false;
	}

	CUresult err = CUDA_SUCCESS;
	if (!impl_->primaryCtxRetained) {
		err = impl_->api.cuInit(0);
		if (err != CUDA_SUCCESS) {
			obs_log(LOG_WARNING, "cuInit failed (%d: %s)", err, impl_->api.errStr(err));
			return false;
		}
		err = impl_->api.cuDeviceGet(&impl_->device, 0);
		if (err != CUDA_SUCCESS) {
			obs_log(LOG_WARNING, "cuDeviceGet(0) failed (%d: %s)", err, impl_->api.errStr(err));
			return false;
		}
		err = impl_->api.cuDevicePrimaryCtxRetain(&impl_->primaryCtx, impl_->device);
		if (err != CUDA_SUCCESS) {
			obs_log(LOG_WARNING, "cuDevicePrimaryCtxRetain failed (%d: %s)", err,
				impl_->api.errStr(err));
			return false;
		}
		impl_->primaryCtxRetained = true;
	}
	if (!impl_->bindCurrentCtx()) {
		shutdown();
		return false;
	}

	if (!impl_->renderStream &&
	    impl_->api.cuStreamCreate(&impl_->renderStream, kCuStreamNonBlocking) != CUDA_SUCCESS) {
		obs_log(LOG_WARNING, "cuStreamCreate(render) failed");
		shutdown();
		return false;
	}
	if (impl_->api.cuStreamCreate(&impl_->workerStream, kCuStreamNonBlocking) != CUDA_SUCCESS) {
		obs_log(LOG_WARNING, "cuStreamCreate(worker) failed");
		shutdown();
		return false;
	}

	char jitErrBuf[2048] = {0};
	CUjit_option jitOpts[2] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
	void *jitVals[2] = {jitErrBuf, reinterpret_cast<void *>(static_cast<uintptr_t>(sizeof(jitErrBuf) - 1))};
	err = impl_->api.cuModuleLoadDataEx(&impl_->ptxModule, kPrePostPtx, 2, jitOpts, jitVals);
	if (err != CUDA_SUCCESS) {
		obs_log(LOG_WARNING, "PTX JIT load failed (%d: %s): %s", err, impl_->api.errStr(err), jitErrBuf);
		shutdown();
		return false;
	}
	if (impl_->api.cuModuleGetFunction(&impl_->fnPreprocess, impl_->ptxModule, "rgba8_to_nchw_rgb_f32") !=
		    CUDA_SUCCESS ||
	    impl_->api.cuModuleGetFunction(&impl_->fnPostprocess, impl_->ptxModule, "pha_f32_to_inv_mask_u8") !=
		    CUDA_SUCCESS) {
		obs_log(LOG_WARNING, "cuModuleGetFunction failed");
		shutdown();
		return false;
	}

	auto allocZero = [&](CUdeviceptr &outPtr, size_t bytes) -> bool {
		if (impl_->api.cuMemAlloc(&outPtr, bytes) != CUDA_SUCCESS) {
			return false;
		}
		return impl_->api.cuMemsetD8Async(outPtr, 0, bytes, impl_->workerStream) == CUDA_SUCCESS;
	};

	const size_t slotBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
	for (int s = 0; s < kInputSlotCount; s++) {
		if (!allocZero(impl_->dInputSlots[s], slotBytes)) {
			obs_log(LOG_WARNING, "Failed allocating CUDA input slot %d (%zu B)", s, slotBytes);
			shutdown();
			return false;
		}
	}

	impl_->srcElems = shapeElements(inputDims[0]);
	if (!allocZero(impl_->dSrc, impl_->srcElems * sizeof(float))) {
		shutdown();
		return false;
	}
	impl_->ortSrc = Ort::Value::CreateTensor<float>(
		impl_->cudaMemInfo, reinterpret_cast<float *>(static_cast<uintptr_t>(impl_->dSrc)), impl_->srcElems,
		inputDims[0].data(), inputDims[0].size());

	impl_->hasRecurrent = hasRecurrentStates;
	impl_->downsampleRatio = downsampleRatio;
	impl_->stateParity = 0;

	if (hasRecurrentStates) {
		for (int i = 0; i < 4; i++) {
			const auto &sShape = inputDims[static_cast<size_t>(i + 1)];
			const size_t elems = shapeElements(sShape);
			impl_->stateBytes[i] = elems * sizeof(float);
			for (int p = 0; p < 2; p++) {
				if (!allocZero(impl_->dStates[p][i], impl_->stateBytes[i])) {
					shutdown();
					return false;
				}
				impl_->ortStates[p][i] = Ort::Value::CreateTensor<float>(
					impl_->cudaMemInfo,
					reinterpret_cast<float *>(static_cast<uintptr_t>(impl_->dStates[p][i])), elems,
					sShape.data(), sShape.size());
			}
		}

		if (!allocZero(impl_->dRatio, sizeof(float))) {
			shutdown();
			return false;
		}
		if (impl_->api.cuMemcpyHtoDAsync(impl_->dRatio, &impl_->downsampleRatio, sizeof(float),
						 impl_->workerStream) != CUDA_SUCCESS) {
			shutdown();
			return false;
		}
		impl_->ortRatio = Ort::Value::CreateTensor<float>(
			impl_->cudaMemInfo, reinterpret_cast<float *>(static_cast<uintptr_t>(impl_->dRatio)), 1,
			inputDims[5].data(), inputDims[5].size());
	}

	impl_->phaElems = shapeElements(outputDims[0]);
	if (!allocZero(impl_->dPha, impl_->phaElems * sizeof(float))) {
		shutdown();
		return false;
	}
	impl_->ortPha = Ort::Value::CreateTensor<float>(
		impl_->cudaMemInfo, reinterpret_cast<float *>(static_cast<uintptr_t>(impl_->dPha)), impl_->phaElems,
		outputDims[0].data(), outputDims[0].size());

	impl_->maskBytes = static_cast<size_t>(width) * static_cast<size_t>(height);
	if (!allocZero(impl_->dMaskScratch, impl_->maskBytes) ||
	    !allocZero(impl_->dMaskPublished, impl_->maskBytes) || !allocZero(impl_->dPrevMask, impl_->maskBytes)) {
		shutdown();
		return false;
	}
	if (impl_->api.cuStreamSynchronize(impl_->workerStream) != CUDA_SUCCESS) {
		shutdown();
		return false;
	}

	const bool glDisabled = (impl_->api.cuGraphicsGLRegisterImage == nullptr) || impl_->hasAnyGlHandles();
	width_ = width;
	height_ = height;
	glInteropDisabled_.store(glDisabled, std::memory_order_release);
	kernelsReady_.store(true, std::memory_order_release);
	initialized_.store(true, std::memory_order_release);

	obs_log(LOG_INFO, "CUDA fp32 GPU pipeline initialized (%ux%u, recurrent=%s, gl_symbols=%s)", width_,
		height_, hasRecurrentStates ? "true" : "false", glDisabled ? "absent" : "present");
	return true;
}

void CudaGpuPipeline::resetRecurrentAndTemporalState()
{
	if (!isInitialized() || !impl_->bindCurrentCtx()) {
		return;
	}
	if (impl_->hasRecurrent) {
		for (int p = 0; p < 2; p++) {
			for (int i = 0; i < 4; i++) {
				if (impl_->dStates[p][i] != 0) {
					impl_->api.cuMemsetD8Async(impl_->dStates[p][i], 0, impl_->stateBytes[i],
								   impl_->workerStream);
				}
			}
		}
		impl_->stateParity = 0;
	}
	if (impl_->dPrevMask != 0) {
		impl_->api.cuMemsetD8Async(impl_->dPrevMask, 255, impl_->maskBytes, impl_->workerStream);
	}
	impl_->hasPrevMask = false;
	impl_->api.cuStreamSynchronize(impl_->workerStream);
}

CudaGpuPipeline::GlInteropStatus CudaGpuPipeline::captureGlTextureToSlotOnGraphicsThread(gs_texture_t *gsTex,
											 int slotIdx)
{
	if (!isInitialized() || isGlInteropDisabled() || !gsTex || slotIdx < 0 || slotIdx >= kInputSlotCount) {
		return GlInteropStatus::FallbackReleased;
	}
	if (gs_get_device_type() != GS_DEVICE_OPENGL) {
		if (!impl_->loggedGlInteropFallback) {
			impl_->loggedGlInteropFallback = true;
			obs_log(LOG_WARNING,
				"OBS device type is not OpenGL; using downscaled stage-surface + CUDA IoBinding fallback");
		}
		glInteropDisabled_.store(true, std::memory_order_release);
		return GlInteropStatus::FallbackReleased;
	}
	if (!impl_->bindCurrentCtx()) {
		return GlInteropStatus::FallbackReleased;
	}

	void *rawObj = gs_texture_get_obj(gsTex);
	if (!rawObj) {
		return GlInteropStatus::FallbackReleased;
	}
	const GLuint glId = *static_cast<const GLuint *>(rawObj);
	const uint32_t texW = gs_texture_get_width(gsTex);
	const uint32_t texH = gs_texture_get_height(gsTex);
	if (glId == 0 || texW != width_ || texH != height_) {
		return GlInteropStatus::FallbackReleased;
	}

	if (impl_->glInputRes == nullptr || impl_->glInputGsTex != gsTex || impl_->glInputId != glId ||
	    impl_->glInputW != texW || impl_->glInputH != texH) {
		if (impl_->glInputRes) {
			if (!impl_->releaseOneGlResource(impl_->glInputRes, impl_->glInputMapped)) {
				impl_->pinnedInputEntries.push_back({impl_->glInputRes, impl_->glInputMapped});
				impl_->glInputRes = nullptr;
				impl_->glInputMapped = false;
				glInteropDisabled_.store(true, std::memory_order_release);
				return GlInteropStatus::FallbackOwnershipPinned;
			}
		}
		const CUresult regErr = impl_->api.cuGraphicsGLRegisterImage(&impl_->glInputRes, glId, kGlTexture2D,
									     kCuGraphicsRegisterFlagsReadOnly);
		if (regErr != CUDA_SUCCESS) {
			if (!impl_->loggedGlInteropFallback) {
				impl_->loggedGlInteropFallback = true;
				obs_log(LOG_WARNING,
					"CUDA-GL interop unavailable on current display backend (cuGraphicsGLRegisterImage=%d: %s); falling back to pre-downscaled stage-surface + CUDA IoBinding",
					regErr, impl_->api.errStr(regErr));
			}
			glInteropDisabled_.store(true, std::memory_order_release);
			return GlInteropStatus::FallbackReleased;
		}
		impl_->glInputMapped = false;
		impl_->api.cuGraphicsResourceSetMapFlags(impl_->glInputRes, kCuGraphicsMapResourceFlagsReadOnly);
		impl_->glInputGsTex = gsTex;
		impl_->glInputId = glId;
		impl_->glInputW = texW;
		impl_->glInputH = texH;
	}

	CUresult err = impl_->api.cuGraphicsMapResources(1, &impl_->glInputRes, impl_->renderStream);
	if (err != CUDA_SUCCESS) {
		if (!impl_->loggedGlInteropFallback) {
			impl_->loggedGlInteropFallback = true;
			obs_log(LOG_WARNING,
				"cuGraphicsMapResources(input) failed (%d: %s); switching to stage-surface + CUDA IoBinding fallback",
				err, impl_->api.errStr(err));
		}
		const bool released = impl_->releaseOneGlResource(impl_->glInputRes, impl_->glInputMapped);
		glInteropDisabled_.store(true, std::memory_order_release);
		if (!released) {
			impl_->pinnedInputEntries.push_back({impl_->glInputRes, impl_->glInputMapped});
			impl_->glInputRes = nullptr;
			impl_->glInputMapped = false;
			return GlInteropStatus::FallbackOwnershipPinned;
		}
		return GlInteropStatus::FallbackReleased;
	}
	impl_->glInputMapped = true;

	CUarray mappedArr = nullptr;
	err = impl_->api.cuGraphicsSubResourceGetMappedArray(&mappedArr, impl_->glInputRes, 0, 0);
	if (err == CUDA_SUCCESS && mappedArr == nullptr) {
		err = -1;
	}
	if (err == CUDA_SUCCESS) {
		CUDA_MEMCPY2D_v2 copyDesc{};
		copyDesc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		copyDesc.srcArray = mappedArr;
		copyDesc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		copyDesc.dstDevice = impl_->dInputSlots[static_cast<size_t>(slotIdx)];
		copyDesc.dstPitch = static_cast<size_t>(width_) * 4u;
		copyDesc.WidthInBytes = static_cast<size_t>(width_) * 4u;
		copyDesc.Height = static_cast<size_t>(height_);
		err = impl_->api.cuMemcpy2DAsync(&copyDesc, impl_->renderStream);
	}

	const CUresult unmapErr = impl_->api.cuGraphicsUnmapResources(1, &impl_->glInputRes, impl_->renderStream);
	if (unmapErr == CUDA_SUCCESS) {
		impl_->glInputMapped = false;
	}
	const CUresult syncErr = impl_->api.cuStreamSynchronize(impl_->renderStream);
	if (err != CUDA_SUCCESS || unmapErr != CUDA_SUCCESS || syncErr != CUDA_SUCCESS) {
		if (!impl_->loggedGlInteropFallback) {
			impl_->loggedGlInteropFallback = true;
			obs_log(LOG_WARNING,
				"CUDA-GL input transfer/sync failed (copy=%d unmap=%d sync=%d); switching to stage-surface + CUDA IoBinding fallback",
				err, unmapErr, syncErr);
		}
		const bool released = impl_->releaseOneGlResource(impl_->glInputRes, impl_->glInputMapped);
		glInteropDisabled_.store(true, std::memory_order_release);
		if (!released) {
			impl_->pinnedInputEntries.push_back({impl_->glInputRes, impl_->glInputMapped});
			impl_->glInputRes = nullptr;
			impl_->glInputMapped = false;
			return GlInteropStatus::FallbackOwnershipPinned;
		}
		return GlInteropStatus::FallbackReleased;
	}
	impl_->slotIsBgra[static_cast<size_t>(slotIdx)] = false;
	return GlInteropStatus::Ok;
}

CudaGpuPipeline::GlInteropStatus CudaGpuPipeline::uploadMaskToGlTextureOnGraphicsThread(gs_texture_t *gsAlphaTex)
{
	if (!isInitialized() || isGlInteropDisabled() || !gsAlphaTex) {
		return GlInteropStatus::FallbackReleased;
	}
	if (gs_get_device_type() != GS_DEVICE_OPENGL || !impl_->bindCurrentCtx()) {
		return GlInteropStatus::FallbackReleased;
	}
	void *rawObj = gs_texture_get_obj(gsAlphaTex);
	if (!rawObj) {
		return GlInteropStatus::FallbackReleased;
	}
	const GLuint glId = *static_cast<const GLuint *>(rawObj);
	const uint32_t texW = gs_texture_get_width(gsAlphaTex);
	const uint32_t texH = gs_texture_get_height(gsAlphaTex);
	if (glId == 0 || texW != width_ || texH != height_) {
		return GlInteropStatus::FallbackReleased;
	}

	if (impl_->glAlphaRes == nullptr || impl_->glAlphaGsTex != gsAlphaTex || impl_->glAlphaId != glId ||
	    impl_->glAlphaW != texW || impl_->glAlphaH != texH) {
		if (impl_->glAlphaRes) {
			if (!impl_->releaseOneGlResource(impl_->glAlphaRes, impl_->glAlphaMapped)) {
				impl_->pinnedAlphaEntries.push_back({impl_->glAlphaRes, impl_->glAlphaMapped});
				impl_->glAlphaRes = nullptr;
				impl_->glAlphaMapped = false;
				return GlInteropStatus::FallbackOwnershipPinned;
			}
		}
		const CUresult regErr = impl_->api.cuGraphicsGLRegisterImage(
			&impl_->glAlphaRes, glId, kGlTexture2D, kCuGraphicsRegisterFlagsWriteDiscard);
		if (regErr != CUDA_SUCCESS) {
			return GlInteropStatus::FallbackReleased;
		}
		impl_->glAlphaMapped = false;
		impl_->api.cuGraphicsResourceSetMapFlags(impl_->glAlphaRes, kCuGraphicsMapResourceFlagsWriteDiscard);
		impl_->glAlphaGsTex = gsAlphaTex;
		impl_->glAlphaId = glId;
		impl_->glAlphaW = texW;
		impl_->glAlphaH = texH;
	}

	CUresult err = impl_->api.cuGraphicsMapResources(1, &impl_->glAlphaRes, impl_->renderStream);
	if (err != CUDA_SUCCESS) {
		const bool released = impl_->releaseOneGlResource(impl_->glAlphaRes, impl_->glAlphaMapped);
		if (!released) {
			impl_->pinnedAlphaEntries.push_back({impl_->glAlphaRes, impl_->glAlphaMapped});
			impl_->glAlphaRes = nullptr;
			impl_->glAlphaMapped = false;
			return GlInteropStatus::FallbackOwnershipPinned;
		}
		return GlInteropStatus::FallbackReleased;
	}
	impl_->glAlphaMapped = true;

	CUarray mappedArr = nullptr;
	err = impl_->api.cuGraphicsSubResourceGetMappedArray(&mappedArr, impl_->glAlphaRes, 0, 0);
	if (err == CUDA_SUCCESS && mappedArr == nullptr) {
		err = -1;
	}
	if (err == CUDA_SUCCESS) {
		CUDA_MEMCPY2D_v2 copyDesc{};
		copyDesc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		copyDesc.srcDevice = impl_->dMaskPublished;
		copyDesc.srcPitch = static_cast<size_t>(width_);
		copyDesc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		copyDesc.dstArray = mappedArr;
		copyDesc.WidthInBytes = static_cast<size_t>(width_);
		copyDesc.Height = static_cast<size_t>(height_);
		err = impl_->api.cuMemcpy2DAsync(&copyDesc, impl_->renderStream);
	}

	const CUresult unmapErr = impl_->api.cuGraphicsUnmapResources(1, &impl_->glAlphaRes, impl_->renderStream);
	if (unmapErr == CUDA_SUCCESS) {
		impl_->glAlphaMapped = false;
	}
	const CUresult syncErr = impl_->api.cuStreamSynchronize(impl_->renderStream);
	if (err != CUDA_SUCCESS || unmapErr != CUDA_SUCCESS || syncErr != CUDA_SUCCESS) {
		const bool released = impl_->releaseOneGlResource(impl_->glAlphaRes, impl_->glAlphaMapped);
		if (!released) {
			impl_->pinnedAlphaEntries.push_back({impl_->glAlphaRes, impl_->glAlphaMapped});
			impl_->glAlphaRes = nullptr;
			impl_->glAlphaMapped = false;
			return GlInteropStatus::FallbackOwnershipPinned;
		}
		return GlInteropStatus::FallbackReleased;
	}
	return GlInteropStatus::Ok;
}

bool CudaGpuPipeline::uploadHostFrameToSlot(const uint8_t *hostData, uint32_t hostLinesize, int slotIdx, bool isBgra)
{
	if (!isInitialized() || !hostData || slotIdx < 0 || slotIdx >= kInputSlotCount || !impl_->bindCurrentCtx()) {
		return false;
	}
	CUDA_MEMCPY2D_v2 copyDesc{};
	copyDesc.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyDesc.srcHost = hostData;
	copyDesc.srcPitch = static_cast<size_t>(hostLinesize);
	copyDesc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyDesc.dstDevice = impl_->dInputSlots[static_cast<size_t>(slotIdx)];
	copyDesc.dstPitch = static_cast<size_t>(width_) * 4u;
	copyDesc.WidthInBytes = static_cast<size_t>(width_) * 4u;
	copyDesc.Height = static_cast<size_t>(height_);

	const CUresult err = impl_->api.cuMemcpy2DAsync(&copyDesc, impl_->workerStream);
	if (err != CUDA_SUCCESS) {
		return false;
	}
	impl_->slotIsBgra[static_cast<size_t>(slotIdx)] = isBgra;
	return true;
}

bool CudaGpuPipeline::runInferenceOnSlot(Ort::Session &session,
					 const std::vector<Ort::AllocatedStringPtr> &inputNames,
					 const std::vector<Ort::AllocatedStringPtr> &outputNames, int slotIdx,
					 bool enableThreshold, float threshold, float temporalSmoothFactor,
					 uint8_t *copyMaskToHost)
{
	if (!isInitialized() || !hasKernels() || slotIdx < 0 || slotIdx >= kInputSlotCount ||
	    !impl_->bindCurrentCtx()) {
		return false;
	}

	// 1. Launch GPU RGBA/BGRA -> NCHW RGB fp32 [0,1] preprocess into dSrc
	CUdeviceptr slotPtr = impl_->dInputSlots[static_cast<size_t>(slotIdx)];
	CUdeviceptr dstPtr = impl_->dSrc;
	uint32_t w = width_;
	uint32_t h = height_;
	uint32_t swapRb = impl_->slotIsBgra[static_cast<size_t>(slotIdx)] ? 1u : 0u;
	void *preArgs[5] = {&slotPtr, &dstPtr, &w, &h, &swapRb};

	const unsigned int blockX = 16;
	const unsigned int blockY = 16;
	const unsigned int gridX = (w + blockX - 1) / blockX;
	const unsigned int gridY = (h + blockY - 1) / blockY;

	CUresult err = impl_->api.cuLaunchKernel(impl_->fnPreprocess, gridX, gridY, 1, blockX, blockY, 1, 0,
						 impl_->workerStream, preArgs, nullptr);
	if (err != CUDA_SUCCESS) {
		obs_log(LOG_ERROR, "fnPreprocess launch failed (%d: %s)", err, impl_->api.errStr(err));
		return false;
	}
	impl_->api.cuStreamSynchronize(impl_->workerStream);

	// 2. Bind CUDA-resident fp32 inputs & outputs via Ort::IoBinding (ping-pong r1..r4, zero H2D/D2H)
	Ort::IoBinding ioBinding{session};
	ioBinding.BindInput(inputNames[0].get(), impl_->ortSrc);
	const int inParity = impl_->stateParity;
	const int outParity = 1 - inParity;
	if (impl_->hasRecurrent) {
		for (int i = 0; i < 4; i++) {
			ioBinding.BindInput(inputNames[static_cast<size_t>(i + 1)].get(),
					    impl_->ortStates[static_cast<size_t>(inParity)][static_cast<size_t>(i)]);
			ioBinding.BindOutput(outputNames[static_cast<size_t>(i + 1)].get(),
					     impl_->ortStates[static_cast<size_t>(outParity)][static_cast<size_t>(i)]);
		}
		ioBinding.BindInput(inputNames[5].get(), impl_->ortRatio);
	}
	ioBinding.BindOutput(outputNames[0].get(), impl_->ortPha);

	ioBinding.SynchronizeInputs();
	session.Run(Ort::RunOptions{nullptr}, ioBinding);
	ioBinding.SynchronizeOutputs();

	if (impl_->hasRecurrent) {
		impl_->stateParity = outParity;
	}

	// 3. Launch GPU pha fp32 -> inverted/thresholded (+ optional EMA smoothed) R8 mask into dMaskScratch
	CUdeviceptr phaPtr = impl_->dPha;
	CUdeviceptr maskPtr = impl_->dMaskScratch;
	CUdeviceptr prevPtr = impl_->dPrevMask;
	uint32_t totalPixels = w * h;
	uint32_t enableThreshU32 = enableThreshold ? 1u : 0u;
	uint32_t threshU8 = static_cast<uint32_t>(std::clamp(threshold * 255.0f, 0.0f, 255.0f));

	float effSmooth = temporalSmoothFactor;
	if (enableThreshold) {
		effSmooth = std::max(effSmooth, threshold);
	}
	const bool applySmooth = impl_->hasPrevMask && (temporalSmoothFactor > 0.0f && temporalSmoothFactor < 1.0f);
	uint32_t useSmoothU32 = applySmooth ? 1u : 0u;

	void *postArgs[8] = {&phaPtr, &maskPtr, &prevPtr, &totalPixels, &enableThreshU32, &threshU8, &useSmoothU32,
			     &effSmooth};
	const unsigned int postBlock = 256;
	const unsigned int postGrid = (totalPixels + postBlock - 1) / postBlock;

	err = impl_->api.cuLaunchKernel(impl_->fnPostprocess, postGrid, 1, 1, postBlock, 1, 1, 0, impl_->workerStream,
					postArgs, nullptr);
	if (err != CUDA_SUCCESS) {
		obs_log(LOG_ERROR, "fnPostprocess launch failed (%d: %s)", err, impl_->api.errStr(err));
		return false;
	}

	if (copyMaskToHost != nullptr) {
		err = impl_->api.cuMemcpyDtoHAsync(copyMaskToHost, impl_->dMaskScratch, impl_->maskBytes,
						   impl_->workerStream);
		if (err != CUDA_SUCCESS) {
			return false;
		}
	}

	if (impl_->api.cuStreamSynchronize(impl_->workerStream) != CUDA_SUCCESS) {
		return false;
	}
	impl_->hasPrevMask = true;
	return true;
}

bool CudaGpuPipeline::publishScratchMask()
{
	if (!isInitialized() || !impl_->bindCurrentCtx()) {
		return false;
	}
	const CUresult err = impl_->api.cuMemcpyDtoDAsync(impl_->dMaskPublished, impl_->dMaskScratch, impl_->maskBytes,
							  impl_->workerStream);
	if (err != CUDA_SUCCESS) {
		return false;
	}
	return impl_->api.cuStreamSynchronize(impl_->workerStream) == CUDA_SUCCESS;
}

bool CudaGpuPipeline::downloadPublishedMaskToHost(uint8_t *hostDst)
{
	if (!isInitialized() || !hostDst || !impl_->bindCurrentCtx()) {
		return false;
	}
	const CUresult err =
		impl_->api.cuMemcpyDtoHAsync(hostDst, impl_->dMaskPublished, impl_->maskBytes, impl_->renderStream);
	if (err != CUDA_SUCCESS) {
		return false;
	}
	return impl_->api.cuStreamSynchronize(impl_->renderStream) == CUDA_SUCCESS;
}

#else

struct CudaGpuPipeline::Impl {};

CudaGpuPipeline::CudaGpuPipeline() = default;
CudaGpuPipeline::~CudaGpuPipeline() = default;

bool CudaGpuPipeline::initialize(uint32_t, uint32_t, bool, float, const std::vector<std::vector<int64_t>> &,
				 const std::vector<std::vector<int64_t>> &)
{
	return false;
}

bool CudaGpuPipeline::releaseGlResourcesOnGraphicsThread()
{
	return true;
}
bool CudaGpuPipeline::releaseGlInputResourceOnGraphicsThread()
{
	return true;
}
bool CudaGpuPipeline::releaseGlAlphaResourceOnGraphicsThread()
{
	return true;
}
void CudaGpuPipeline::shutdown() {}
CudaGpuPipeline::GlInteropStatus CudaGpuPipeline::captureGlTextureToSlotOnGraphicsThread(gs_texture_t *, int)
{
	return GlInteropStatus::FallbackReleased;
}
CudaGpuPipeline::GlInteropStatus CudaGpuPipeline::uploadMaskToGlTextureOnGraphicsThread(gs_texture_t *)
{
	return GlInteropStatus::FallbackReleased;
}
bool CudaGpuPipeline::uploadHostFrameToSlot(const uint8_t *, uint32_t, int, bool)
{
	return false;
}
void CudaGpuPipeline::resetRecurrentAndTemporalState() {}
bool CudaGpuPipeline::runInferenceOnSlot(Ort::Session &, const std::vector<Ort::AllocatedStringPtr> &,
					 const std::vector<Ort::AllocatedStringPtr> &, int, bool, float, float,
					 uint8_t *)
{
	return false;
}
bool CudaGpuPipeline::publishScratchMask()
{
	return false;
}
bool CudaGpuPipeline::downloadPublishedMaskToHost(uint8_t *)
{
	return false;
}

#endif
