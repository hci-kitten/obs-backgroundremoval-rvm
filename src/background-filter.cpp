// SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
// SPDX-FileCopyrightText: 2026 hci-kitten
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "background-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#endif // _WIN32

#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <regex>
#include <thread>

#include "plugin-support.h"
#include "models/ModelSINET.hpp"
#include "models/ModelMediapipe.hpp"
#include "models/ModelSelfie.hpp"
#include "models/ModelSelfieMulticlass.hpp"
#include "models/ModelRVM.hpp"
#include "models/ModelPPHumanSeg.hpp"
#include "models/ModelTCMonoDepth.hpp"
#include "FilterData.hpp"
#include "ort-utils/model-downloader.hpp"
#include "ort-utils/ort-session-utils.hpp"
#include "obs-utils/obs-utils.hpp"
#include "consts.h"
#include "update-checker/update-checker.h"

namespace {

constexpr int kInputSlotCount = 3;

struct FilterParamsSnapshot {
	bool enableThreshold = true;
	float threshold = 0.5f;
	float contourFilter = 0.05f;
	float smoothContour = 0.5f;
	float feather = 0.0f;
	int maskExpansion = 0;
	float temporalSmoothFactor = 0.0f;
	float imageSimilarityThreshold = 35.0f;
	bool enableImageSimilarity = true;
};

struct FrameSlotMeta {
	bool isBgra = false;
	uint32_t sourceW = 0;
	uint32_t sourceH = 0;
	uint32_t netW = 0;
	uint32_t netH = 0;
	uint64_t epoch = 0;
};

} // namespace

struct background_removal_filter : public filter_data, public std::enable_shared_from_this<background_removal_filter> {
	bool enableThreshold = true;
	bool stopWhenSourceIsInactive = true;
	float threshold = 0.5f;
	cv::Scalar backgroundColor{0, 0, 0, 0};
	float contourFilter = 0.05f;
	float smoothContour = 0.5f;
	float feather = 0.0f;
	int maskExpansion = 0;

	cv::Mat backgroundMask;
	cv::Mat lastBackgroundMask;
	cv::Mat lastImageBGRA;
	float temporalSmoothFactor = 0.0f;
	float imageSimilarityThreshold = 35.0f;
	bool enableImageSimilarity = true;
	std::atomic<int> maskEveryXFrames{1};
	uint64_t renderFrameCounter = 0;
	int64_t blurBackground = 0;
	bool enableFocalBlur = false;
	float blurFocusPoint = 0.1f;
	float blurFocusDepth = 0.1f;

	gs_effect_t *effect = nullptr;
	gs_effect_t *kawaseBlurEffect = nullptr;

	// Dedicated GPU downscale render target (GS_RGBA at network input size) & stage surface
	gs_texrender_t *inferTexrender = nullptr;
	gs_stagesurf_t *inferStagesurface = nullptr;

	// Persistent output mask texture (reused across frames; updated only when maskGeneration advances)
	gs_texture_t *alphaTexture = nullptr;
	uint32_t alphaTexW = 0;
	uint32_t alphaTexH = 0;
	uint64_t uploadedMaskGen = 0;

	// Published mask state (guarded by outputLock)
	uint64_t publishedMaskGen = 0;
	uint32_t publishedMaskW = 0;
	uint32_t publishedMaskH = 0;

	std::atomic<uint32_t> netInputW{0};
	std::atomic<uint32_t> netInputH{0};

	// Worker thread & 3-slot latest-frame-wins ring buffer (guarded by workerMutex; stopWorker/modelEpoch atomic)
	std::mutex workerMutex;
	std::condition_variable workerCv;
	std::thread workerThread;
	std::atomic<bool> stopWorker{false};
	int renderSlot = -1;
	int latestSlot = -1;
	int busySlot = -1;
	uint64_t droppedByLatest = 0;
	FilterParamsSnapshot paramsSnapshot;
	std::array<FrameSlotMeta, kInputSlotCount> slotMeta{};
	std::array<cv::Mat, kInputSlotCount> cpuInputSlots{};

	// Asynchronous first-use model download state (guarded by workerMutex; modelLoadInFlight/sessionReady atomic)
	bool pendingModelLoad = false;
	uint64_t pendingLoadEpoch = 0;
	std::string pendingLoadModel;
	std::atomic<bool> modelLoadInFlight{false};
	std::atomic<bool> sessionReady{false};

	// Model / session serialization (outer lock in update & worker; NEVER acquired on OBS graphics thread)
	std::mutex modelMutex;
	std::atomic<uint64_t> modelEpoch{0};
	std::atomic<bool> resetRecurrentPending{false};

	// Telemetry on worker thread
	bool loggedFirstFramePath = false;
	uint64_t statsFrames = 0;
	double statsAccumMs = 0.0;

	~background_removal_filter() { obs_log(LOG_INFO, "Background removal filter destructor called"); }
};

static void background_removal_worker(background_removal_filter *tf);

const char *background_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("BackgroundRemoval");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop, const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}

static bool enable_threshold_modified(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
	UNUSED_PARAMETER(p);
	return visible_on_bool(ppts, settings, "enable_threshold", "threshold_group");
}

static bool enable_focal_blur(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
	UNUSED_PARAMETER(p);
	return visible_on_bool(ppts, settings, "enable_focal_blur", "focal_blur_group");
}

static bool enable_image_similarity(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
	UNUSED_PARAMETER(p);
	return visible_on_bool(ppts, settings, "enable_image_similarity", "image_similarity_threshold");
}

static bool enable_advanced_settings(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
	const bool enabled = obs_data_get_bool(settings, "advanced");
	p = obs_properties_get(ppts, "blur_background");
	obs_property_set_visible(p, true);

	for (const char *prop_name :
	     {"model_select", "useGPU", "mask_every_x_frames", "numThreads", "enable_focal_blur", "enable_threshold",
	      "threshold_group", "focal_blur_group", "temporal_smooth_factor", "image_similarity_threshold",
	      "enable_image_similarity", "mask_expansion"}) {
		p = obs_properties_get(ppts, prop_name);
		obs_property_set_visible(p, enabled);
	}

	if (enabled) {
		enable_threshold_modified(ppts, p, settings);
		enable_focal_blur(ppts, p, settings);
		enable_image_similarity(ppts, p, settings);
	}

	return true;
}

obs_properties_t *background_filter_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *p_advanced = obs_properties_add_bool(props, "advanced", obs_module_text("Advanced"));
	obs_property_set_modified_callback(p_advanced, enable_advanced_settings);

	obs_properties_add_bool(props, "stop_when_source_is_inactive", obs_module_text("StopWhenSourceIsInactive"));

	obs_property_t *p_enable_threshold =
		obs_properties_add_bool(props, "enable_threshold", obs_module_text("EnableThreshold"));
	obs_property_set_modified_callback(p_enable_threshold, enable_threshold_modified);

	// Threshold props group
	obs_properties_t *threshold_props = obs_properties_create();

	obs_properties_add_float_slider(threshold_props, "threshold", obs_module_text("Threshold"), 0.0, 1.0, 0.025);

	obs_properties_add_float_slider(threshold_props, "contour_filter", obs_module_text("ContourFilter"), 0.0, 1.0,
					0.025);

	obs_properties_add_float_slider(threshold_props, "smooth_contour", obs_module_text("SmoothSilhouette"), 0.0,
					1.0, 0.05);

	obs_properties_add_float_slider(threshold_props, "feather", obs_module_text("FeatherBlendSilhouette"), 0.0, 1.0,
					0.05);

	obs_properties_add_int_slider(threshold_props, "mask_expansion", obs_module_text("MaskExpansion"), -5, 5, 1);

	obs_properties_add_group(props, "threshold_group", obs_module_text("ThresholdGroup"), OBS_GROUP_NORMAL,
				 threshold_props);

	/* GPU, CPU and performance Props */
	obs_property_t *p_use_gpu = obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
							    OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUCUDA"), USEGPU_CUDA);
#endif
#ifdef HAVE_ONNXRUNTIME_ROCM_EP
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUROCM"), USEGPU_ROCM);
#endif
#ifdef HAVE_ONNXRUNTIME_MIGRAPHX_EP
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUMIGRAPHX"), USEGPU_MIGRAPHX);
#endif
#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
	obs_property_list_add_string(p_use_gpu, obs_module_text("TENSORRT"), USEGPU_TENSORRT);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	obs_properties_add_int(props, "mask_every_x_frames", obs_module_text("CalculateMaskEveryXFrame"), 1, 300, 1);
	obs_properties_add_int_slider(props, "numThreads", obs_module_text("NumThreads"), 0, 8, 1);

	/* Model selection Props */
	obs_property_t *p_model_select = obs_properties_add_list(props, "model_select",
								 obs_module_text("SegmentationModel"),
								 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_model_select, obs_module_text("SINet"), MODEL_SINET);
	obs_property_list_add_string(p_model_select, obs_module_text("MediaPipe"), MODEL_MEDIAPIPE);
	obs_property_list_add_string(p_model_select, obs_module_text("Selfie Segmentation"), MODEL_SELFIE);
	obs_property_list_add_string(p_model_select, obs_module_text("Selfie Multiclass"), MODEL_SELFIE_MULTICLASS);
	obs_property_list_add_string(p_model_select, obs_module_text("PPHumanSeg"), MODEL_PPHUMANSEG);
	obs_property_list_add_string(p_model_select, obs_module_text("Robust Video Matting"), MODEL_RVM);
	obs_property_list_add_string(p_model_select, "Robust Video Matting HQ (ResNet50, 720p)", MODEL_RVM_RESNET);
	obs_property_list_add_string(p_model_select, obs_module_text("TCMonoDepth"), MODEL_DEPTH_TCMONODEPTH);

	obs_properties_add_float_slider(props, "temporal_smooth_factor", obs_module_text("TemporalSmoothFactor"), 0.0,
					1.0, 0.01);

	obs_property_t *p_enable_image_similarity =
		obs_properties_add_bool(props, "enable_image_similarity", obs_module_text("EnableImageSimilarity"));
	obs_property_set_modified_callback(p_enable_image_similarity, enable_image_similarity);

	obs_properties_add_float_slider(props, "image_similarity_threshold",
					obs_module_text("ImageSimilarityThreshold"), 0.0, 100.0, 1.0);

	/* Background Blur Props */
	obs_properties_add_int_slider(props, "blur_background", obs_module_text("BlurBackgroundFactor0NoBlurUseColor"),
				      0, 20, 1);

	obs_property_t *p_enable_focal_blur =
		obs_properties_add_bool(props, "enable_focal_blur", obs_module_text("EnableFocalBlur"));
	obs_property_set_modified_callback(p_enable_focal_blur, enable_focal_blur);

	obs_properties_t *focal_blur_props = obs_properties_create();

	obs_properties_add_float_slider(focal_blur_props, "blur_focus_point", obs_module_text("BlurFocusPoint"), 0.0,
					1.0, 0.05);
	obs_properties_add_float_slider(focal_blur_props, "blur_focus_depth", obs_module_text("BlurFocusDepth"), 0.0,
					0.3, 0.02);

	obs_properties_add_group(props, "focal_blur_group", obs_module_text("FocalBlurGroup"), OBS_GROUP_NORMAL,
				 focal_blur_props);

	std::string basic_info = std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	if (get_latest_version() != nullptr) {
		basic_info += std::regex_replace(PLUGIN_INFO_TEMPLATE_UPDATE_AVAILABLE, std::regex("%1"),
						 get_latest_version());
	}
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void background_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "advanced", false);
	obs_data_set_default_bool(settings, "stop_when_source_is_inactive", true);
	obs_data_set_default_bool(settings, "enable_threshold", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_double(settings, "contour_filter", 0.05);
	obs_data_set_default_double(settings, "smooth_contour", 0.5);
	obs_data_set_default_double(settings, "feather", 0.0);
	obs_data_set_default_int(settings, "mask_expansion", 0);
	obs_data_set_default_string(settings, "model_select", MODEL_MEDIAPIPE);
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
	obs_data_set_default_int(settings, "mask_every_x_frames", 1);
	obs_data_set_default_int(settings, "blur_background", 0);
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_double(settings, "temporal_smooth_factor", 0.0);
	obs_data_set_default_double(settings, "image_similarity_threshold", 35.0);
	obs_data_set_default_bool(settings, "enable_image_similarity", true);
	obs_data_set_default_bool(settings, "enable_focal_blur", false);
	obs_data_set_default_double(settings, "blur_focus_point", 0.1);
	obs_data_set_default_double(settings, "blur_focus_depth", 0.1);
}

static bool initializeSessionFromPathLocked(background_removal_filter *tf, const std::string &resolvedPath)
{
	const int ortSessionResult = createOrtSession(tf, resolvedPath);
	if (ortSessionResult != OBS_BGREMOVAL_ORT_SESSION_SUCCESS) {
		obs_log(LOG_ERROR, "Failed to create ONNXRuntime session. Error code: %d", ortSessionResult);
		tf->session.reset();
		tf->sessionReady.store(false, std::memory_order_release);
		tf->netInputW.store(0, std::memory_order_release);
		tf->netInputH.store(0, std::memory_order_release);
		return false;
	}

	uint32_t inW = 0;
	uint32_t inH = 0;
	tf->model->getNetworkInputSize(tf->inputDims, inW, inH);
	tf->netInputW.store(inW, std::memory_order_release);
	tf->netInputH.store(inH, std::memory_order_release);
	tf->sessionReady.store(true, std::memory_order_release);

	{
		std::lock_guard<std::mutex> outLock(tf->outputLock);
		tf->backgroundMask.release();
		tf->publishedMaskGen = 0;
		tf->publishedMaskW = 0;
		tf->publishedMaskH = 0;
	}
	return true;
}

void background_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Background filter updated");

	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<background_removal_filter> tf = *ptr;
	if (!tf) {
		return;
	}

	const char *rawModel = obs_data_get_string(settings, "model_select");
	const char *rawGpu = obs_data_get_string(settings, "useGPU");
	const std::string reqModel = rawModel ? rawModel : "";
	const std::string reqGpu = rawGpu ? rawGpu : "";
	const uint32_t reqThreads = static_cast<uint32_t>(obs_data_get_int(settings, "numThreads"));

	const bool modelLoadActive = tf->modelLoadInFlight.load(std::memory_order_acquire);
	const bool sessionActive = tf->sessionReady.load(std::memory_order_acquire);
	const bool needModelReload = (!sessionActive && !modelLoadActive) || tf->modelSelection.empty() ||
				     tf->modelSelection != reqModel || tf->useGPU != reqGpu ||
				     tf->numThreads != reqThreads;

	// Fast slider-only update path: update scalar parameters without stalling render or resetting RVM states.
	if (!needModelReload && tf->effect != nullptr && tf->kawaseBlurEffect != nullptr) {
		tf->stopWhenSourceIsInactive = obs_data_get_bool(settings, "stop_when_source_is_inactive");
		tf->enableThreshold = obs_data_get_bool(settings, "enable_threshold");
		tf->threshold = static_cast<float>(obs_data_get_double(settings, "threshold"));
		tf->contourFilter = static_cast<float>(obs_data_get_double(settings, "contour_filter"));
		tf->smoothContour = static_cast<float>(obs_data_get_double(settings, "smooth_contour"));
		tf->maskExpansion = static_cast<int>(obs_data_get_double(settings, "mask_expansion"));
		tf->feather = static_cast<float>(obs_data_get_double(settings, "feather"));
		tf->maskEveryXFrames.store(std::max(1, static_cast<int>(obs_data_get_int(settings, "mask_every_x_frames"))),
					   std::memory_order_relaxed);
		tf->blurBackground = obs_data_get_int(settings, "blur_background");
		tf->enableFocalBlur = obs_data_get_bool(settings, "enable_focal_blur");
		tf->blurFocusPoint = static_cast<float>(obs_data_get_double(settings, "blur_focus_point"));
		tf->blurFocusDepth = static_cast<float>(obs_data_get_double(settings, "blur_focus_depth"));
		tf->temporalSmoothFactor = static_cast<float>(obs_data_get_double(settings, "temporal_smooth_factor"));
		tf->imageSimilarityThreshold =
			static_cast<float>(obs_data_get_double(settings, "image_similarity_threshold"));
		tf->enableImageSimilarity = obs_data_get_bool(settings, "enable_image_similarity");

		{
			std::lock_guard<std::mutex> wlock(tf->workerMutex);
			tf->paramsSnapshot.enableThreshold = tf->enableThreshold;
			tf->paramsSnapshot.threshold = tf->threshold;
			tf->paramsSnapshot.contourFilter = tf->contourFilter;
			tf->paramsSnapshot.smoothContour = tf->smoothContour;
			tf->paramsSnapshot.feather = tf->feather;
			tf->paramsSnapshot.maskExpansion = tf->maskExpansion;
			tf->paramsSnapshot.temporalSmoothFactor = tf->temporalSmoothFactor;
			tf->paramsSnapshot.imageSimilarityThreshold = tf->imageSimilarityThreshold;
			tf->paramsSnapshot.enableImageSimilarity = tf->enableImageSimilarity;
		}
		return;
	}

	// Full model/session/shader initialization:
	// Cancel any in-flight model download immediately via modelEpoch before taking locks.
	uint64_t newEpoch = tf->modelEpoch.load(std::memory_order_acquire);
	if (needModelReload) {
		newEpoch = tf->modelEpoch.fetch_add(1, std::memory_order_acq_rel) + 1;
	}
	tf->isDisabled.store(true, std::memory_order_seq_cst);

	obs_enter_graphics();
	if (tf->alphaTexture) {
		gs_texture_destroy(tf->alphaTexture);
		tf->alphaTexture = nullptr;
		tf->alphaTexW = 0;
		tf->alphaTexH = 0;
		tf->uploadedMaskGen = 0;
	}

	char *effect_path = obs_module_file(EFFECT_PATH);
	gs_effect_destroy(tf->effect);
	tf->effect = gs_effect_create_from_file(effect_path, NULL);
	bfree(effect_path);

	char *kawaseBlurEffectPath = obs_module_file(KAWASE_BLUR_EFFECT_PATH);
	gs_effect_destroy(tf->kawaseBlurEffect);
	tf->kawaseBlurEffect = gs_effect_create_from_file(kawaseBlurEffectPath, NULL);
	bfree(kawaseBlurEffectPath);
	obs_leave_graphics();

	tf->stopWhenSourceIsInactive = obs_data_get_bool(settings, "stop_when_source_is_inactive");
	tf->enableThreshold = obs_data_get_bool(settings, "enable_threshold");
	tf->threshold = static_cast<float>(obs_data_get_double(settings, "threshold"));
	tf->contourFilter = static_cast<float>(obs_data_get_double(settings, "contour_filter"));
	tf->smoothContour = static_cast<float>(obs_data_get_double(settings, "smooth_contour"));
	tf->maskExpansion = static_cast<int>(obs_data_get_double(settings, "mask_expansion"));
	tf->feather = static_cast<float>(obs_data_get_double(settings, "feather"));
	tf->maskEveryXFrames.store(std::max(1, static_cast<int>(obs_data_get_int(settings, "mask_every_x_frames"))),
				   std::memory_order_relaxed);
	tf->renderFrameCounter = 0;
	tf->blurBackground = obs_data_get_int(settings, "blur_background");
	tf->enableFocalBlur = obs_data_get_bool(settings, "enable_focal_blur");
	tf->blurFocusPoint = static_cast<float>(obs_data_get_double(settings, "blur_focus_point"));
	tf->blurFocusDepth = static_cast<float>(obs_data_get_double(settings, "blur_focus_depth"));
	tf->temporalSmoothFactor = static_cast<float>(obs_data_get_double(settings, "temporal_smooth_factor"));
	tf->imageSimilarityThreshold = static_cast<float>(obs_data_get_double(settings, "image_similarity_threshold"));
	tf->enableImageSimilarity = obs_data_get_bool(settings, "enable_image_similarity");

	bool queuedAsyncDownload = false;

	if (needModelReload) {
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		tf->sessionReady.store(false, std::memory_order_release);
		tf->session.reset();
		tf->netInputW.store(0, std::memory_order_release);
		tf->netInputH.store(0, std::memory_order_release);

		tf->modelSelection = reqModel;
		tf->useGPU = reqGpu;
		tf->numThreads = reqThreads;
		tf->loggedFirstFramePath = false;
		tf->statsFrames = 0;
		tf->statsAccumMs = 0.0;
		tf->lastBackgroundMask.release();
		tf->lastImageBGRA.release();

		if (tf->modelSelection == MODEL_SINET) {
			tf->model.reset(new ModelSINET);
		}
		if (tf->modelSelection == MODEL_SELFIE) {
			tf->model.reset(new ModelSelfie);
		}
		if (tf->modelSelection == MODEL_SELFIE_MULTICLASS) {
			tf->model.reset(new ModelSelfieMulticlass);
		}
		if (tf->modelSelection == MODEL_MEDIAPIPE) {
			tf->model.reset(new ModelMediaPipe);
		}
		if (tf->modelSelection == MODEL_RVM) {
			tf->model.reset(new ModelRVM);
		}
		if (tf->modelSelection == MODEL_RVM_RESNET) {
			tf->model.reset(new ModelRVM(true));
		}
		if (tf->modelSelection == MODEL_PPHUMANSEG) {
			tf->model.reset(new ModelPPHumanSeg);
		}
		if (tf->modelSelection == MODEL_DEPTH_TCMONODEPTH) {
			tf->model.reset(new ModelTCMonoDepth);
		}

		{
			std::lock_guard<std::mutex> outLock(tf->outputLock);
			tf->backgroundMask.release();
			tf->publishedMaskGen = 0;
			tf->publishedMaskW = 0;
			tf->publishedMaskH = 0;
		}

		if (isDownloadableModel(tf->modelSelection)) {
			// Always verify/download/load large on-demand models on the worker thread so
			// neither the 107 MiB SHA-256 check nor session creation blocks OBS's UI callback.
			queuedAsyncDownload = true;
			obs_log(LOG_INFO, "Scheduling background verification/load for model %s",
				tf->modelSelection.c_str());
		} else {
			std::string existingPath;
			if (resolveExistingModelPath(tf->modelSelection, existingPath)) {
				if (!initializeSessionFromPathLocked(tf.get(), existingPath)) {
					tf->isDisabled.store(true, std::memory_order_seq_cst);
					tf->model.reset();
					return;
				}
			} else {
				obs_log(LOG_ERROR, "Unable to locate model %s", tf->modelSelection.c_str());
				tf->isDisabled.store(true, std::memory_order_seq_cst);
				tf->model.reset();
				return;
			}
		}
	}

	{
		std::lock_guard<std::mutex> wlock(tf->workerMutex);
		tf->latestSlot = -1;
		tf->paramsSnapshot.enableThreshold = tf->enableThreshold;
		tf->paramsSnapshot.threshold = tf->threshold;
		tf->paramsSnapshot.contourFilter = tf->contourFilter;
		tf->paramsSnapshot.smoothContour = tf->smoothContour;
		tf->paramsSnapshot.feather = tf->feather;
		tf->paramsSnapshot.maskExpansion = tf->maskExpansion;
		tf->paramsSnapshot.temporalSmoothFactor = tf->temporalSmoothFactor;
		tf->paramsSnapshot.imageSimilarityThreshold = tf->imageSimilarityThreshold;
		tf->paramsSnapshot.enableImageSimilarity = tf->enableImageSimilarity;
		if (needModelReload) {
			if (queuedAsyncDownload) {
				tf->pendingModelLoad = true;
				tf->pendingLoadEpoch = newEpoch;
				tf->pendingLoadModel = tf->modelSelection;
				tf->modelLoadInFlight.store(true, std::memory_order_release);
			} else {
				tf->pendingModelLoad = false;
				tf->modelLoadInFlight.store(false, std::memory_order_release);
			}
		}
	}

	obs_log(LOG_INFO, "Background Removal Filter Options:");
	obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
	obs_log(LOG_INFO, "  Model: %s", tf->modelSelection.c_str());
	obs_log(LOG_INFO, "  Inference Device: %s (cudaSession=%s)", tf->useGPU.c_str(),
		tf->cudaSessionActive ? "true" : "false");
	obs_log(LOG_INFO, "  Num Threads: %u", tf->numThreads);
	obs_log(LOG_INFO, "  Enable Threshold: %s", tf->enableThreshold ? "true" : "false");
	obs_log(LOG_INFO, "  Threshold: %f", tf->threshold);
	obs_log(LOG_INFO, "  Contour Filter: %f", tf->contourFilter);
	obs_log(LOG_INFO, "  Smooth Contour: %f", tf->smoothContour);
	obs_log(LOG_INFO, "  Mask Expansion: %d", tf->maskExpansion);
	obs_log(LOG_INFO, "  Feather: %f", tf->feather);
	obs_log(LOG_INFO, "  Mask Every X Frames: %d", tf->maskEveryXFrames.load());
	obs_log(LOG_INFO, "  Enable Image Similarity: %s", tf->enableImageSimilarity ? "true" : "false");
	obs_log(LOG_INFO, "  Image Similarity Threshold: %f", tf->imageSimilarityThreshold);
	obs_log(LOG_INFO, "  Blur Background: %d", static_cast<int>(tf->blurBackground));
	obs_log(LOG_INFO, "  Enable Focal Blur: %s", tf->enableFocalBlur ? "true" : "false");
	obs_log(LOG_INFO, "  Blur Focus Point: %f", tf->blurFocusPoint);
	obs_log(LOG_INFO, "  Blur Focus Depth: %f", tf->blurFocusDepth);
	if (!queuedAsyncDownload) {
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %S", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	tf->isDisabled.store(false, std::memory_order_seq_cst);
	if (queuedAsyncDownload) {
		tf->workerCv.notify_one();
	}
}

void background_filter_activate(void *data)
{
	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<background_removal_filter> tf = *ptr;
	if (tf && tf->stopWhenSourceIsInactive) {
		obs_log(LOG_INFO, "Background filter activated");
		tf->resetRecurrentPending.store(true, std::memory_order_release);
		tf->isDisabled.store(false, std::memory_order_release);
	}
}

void background_filter_deactivate(void *data)
{
	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<background_removal_filter> tf = *ptr;
	if (tf && tf->stopWhenSourceIsInactive) {
		obs_log(LOG_INFO, "Background filter deactivated");
		tf->isDisabled.store(true, std::memory_order_release);
	}
}

/**                   FILTER CORE                     */

void *background_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Background filter created");
	try {
		auto instance = std::make_shared<background_removal_filter>();

		instance->source = source;
		obs_enter_graphics();
		instance->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
		instance->inferTexrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
		obs_leave_graphics();

		std::string instanceName{"background-removal-inference"};
		instance->env.reset(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str()));

		instance->modelSelection = MODEL_MEDIAPIPE;

		auto ptr = new std::shared_ptr<background_removal_filter>(instance);
		background_filter_update(ptr, settings);

		instance->workerThread = std::thread(background_removal_worker, instance.get());

		return ptr;
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "Failed to create background filter: %s", e.what());
		return nullptr;
	}
}

void background_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Background filter destroyed");

	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (ptr) {
		if (*ptr) {
			background_removal_filter *tf = ptr->get();
			tf->isDisabled.store(true, std::memory_order_seq_cst);
			tf->modelEpoch.fetch_add(1, std::memory_order_acq_rel);

			// Stop and join worker thread BEFORE touching graphics context or ORT state.
			{
				std::lock_guard<std::mutex> wlock(tf->workerMutex);
				tf->stopWorker.store(true, std::memory_order_release);
				tf->pendingModelLoad = false;
				tf->latestSlot = -1;
			}
			tf->workerCv.notify_all();
			if (tf->workerThread.joinable()) {
				tf->workerThread.join();
			}

			obs_enter_graphics();
			if (tf->inferTexrender) {
				gs_texrender_destroy(tf->inferTexrender);
				tf->inferTexrender = nullptr;
			}
			if (tf->alphaTexture) {
				gs_texture_destroy(tf->alphaTexture);
				tf->alphaTexture = nullptr;
			}
			if (tf->inferStagesurface) {
				gs_stagesurface_destroy(tf->inferStagesurface);
				tf->inferStagesurface = nullptr;
			}
			gs_texrender_destroy(tf->texrender);
			tf->texrender = nullptr;
			if (tf->stagesurface) {
				gs_stagesurface_destroy(tf->stagesurface);
				tf->stagesurface = nullptr;
			}
			gs_effect_destroy(tf->effect);
			tf->effect = nullptr;
			gs_effect_destroy(tf->kawaseBlurEffect);
			tf->kawaseBlurEffect = nullptr;
			obs_leave_graphics();
		}
		delete ptr;
	}
}

static void processImageForBackground(background_removal_filter *tf, const cv::Mat &imageInput, bool inputIsRGBA,
				      const FilterParamsSnapshot &params, cv::Mat &backgroundMask)
{
	cv::Mat outputImage;
	if (!runFilterModelInference(tf, imageInput, outputImage, inputIsRGBA)) {
		return;
	}

	if (tf->modelSelection == MODEL_SELFIE_MULTICLASS) {
		std::vector<cv::Mat> outputImageSplit;
		cv::split(outputImage, outputImageSplit);
		cv::Mat background, hair, body, face, clothes;
		background = outputImageSplit[0];
		hair = outputImageSplit[1];
		body = outputImageSplit[2];
		face = outputImageSplit[3];
		clothes = outputImageSplit[4];
		outputImage = cv::min(cv::max(cv::max(hair, body), cv::max(face, clothes)), 255 - background);
	}

	backgroundMask = 255 - outputImage;

	if (params.enableThreshold) {
		if (params.threshold < 1.0f) {
			backgroundMask = backgroundMask > static_cast<int>(params.threshold * 255.0f);
		}
	}
}

static void applyThresholdMorphology(cv::Mat &backgroundMask, const FilterParamsSnapshot &params, uint32_t outW,
				     uint32_t outH)
{
	if (backgroundMask.empty()) {
		return;
	}
	if (params.contourFilter > 0.0f && params.contourFilter < 1.0f) {
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(backgroundMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		std::vector<std::vector<cv::Point>> filteredContours;
		const double contourSizeThreshold =
			static_cast<double>(backgroundMask.total()) * static_cast<double>(params.contourFilter);
		for (auto &contour : contours) {
			if (cv::contourArea(contour) > contourSizeThreshold) {
				filteredContours.push_back(contour);
			}
		}
		backgroundMask.setTo(0);
		cv::drawContours(backgroundMask, filteredContours, -1, cv::Scalar(255), -1);
	}

	if (outW > 0 && outH > 0 &&
	    (static_cast<uint32_t>(backgroundMask.cols) != outW ||
	     static_cast<uint32_t>(backgroundMask.rows) != outH)) {
		cv::resize(backgroundMask, backgroundMask, cv::Size(static_cast<int>(outW), static_cast<int>(outH)));
	}

	if (params.smoothContour > 0.0f) {
		backgroundMask = backgroundMask > 128;
	}

	if (params.maskExpansion > 0) {
		cv::erode(backgroundMask, backgroundMask, cv::Mat(), cv::Point(-1, -1), params.maskExpansion);
	} else if (params.maskExpansion < 0) {
		cv::dilate(backgroundMask, backgroundMask, cv::Mat(), cv::Point(-1, -1), -params.maskExpansion);
	}

	if (params.feather > 0.0f) {
		int k_size = static_cast<int>(40 * params.feather);
		k_size += k_size % 2 == 0 ? 1 : 0;
		cv::dilate(backgroundMask, backgroundMask, cv::Mat(), cv::Point(-1, -1), k_size / 3);
		cv::boxFilter(backgroundMask, backgroundMask, backgroundMask.depth(), cv::Size(k_size, k_size));
	}
}

static void background_removal_worker(background_removal_filter *tf)
{
	while (true) {
		bool doModelLoad = false;
		uint64_t loadEpoch = 0;
		std::string loadModel;
		int slot = -1;
		FrameSlotMeta meta{};
		FilterParamsSnapshot params{};
		uint64_t droppedSnapshot = 0;

		{
			std::unique_lock<std::mutex> wlock(tf->workerMutex);
			tf->workerCv.wait(wlock, [&] {
				return tf->stopWorker.load(std::memory_order_acquire) || tf->pendingModelLoad ||
				       (tf->latestSlot != -1 && !tf->isDisabled.load(std::memory_order_acquire));
			});
			if (tf->stopWorker.load(std::memory_order_acquire)) {
				break;
			}
			if (tf->pendingModelLoad) {
				doModelLoad = true;
				loadEpoch = tf->pendingLoadEpoch;
				loadModel = tf->pendingLoadModel;
				tf->pendingModelLoad = false;
			} else {
				slot = tf->latestSlot;
				tf->latestSlot = -1;
				tf->busySlot = slot;
				meta = tf->slotMeta[static_cast<size_t>(slot)];
				params = tf->paramsSnapshot;
				droppedSnapshot = tf->droppedByLatest;
			}
		}

		if (doModelLoad) {
			auto shouldCancel = [tf, loadEpoch]() {
				return tf->stopWorker.load(std::memory_order_acquire) ||
				       tf->modelEpoch.load(std::memory_order_acquire) != loadEpoch;
			};

			try {
				std::string resolvedPath;
				const bool available = ensureModelAvailable(loadModel, resolvedPath, shouldCancel);
				if (available && !shouldCancel()) {
					std::unique_lock<std::mutex> mlock(tf->modelMutex);
					if (!shouldCancel() && tf->model) {
						if (initializeSessionFromPathLocked(tf, resolvedPath)) {
							obs_log(LOG_INFO,
								"Background Removal model ready: %s (cudaSession=%s)",
								loadModel.c_str(),
								tf->cudaSessionActive ? "true" : "false");
						}
					}
				} else if (!shouldCancel()) {
					obs_log(LOG_ERROR,
						"Failed to prepare model %s; filter remains pass-through until re-selected",
						loadModel.c_str());
				}
			} catch (const Ort::Exception &e) {
				obs_log(LOG_ERROR, "ONNXRuntime exception while loading model %s: %s",
					loadModel.c_str(), e.what());
				std::unique_lock<std::mutex> mlock(tf->modelMutex);
				if (tf->modelEpoch.load(std::memory_order_acquire) == loadEpoch) {
					tf->sessionReady.store(false, std::memory_order_release);
					tf->session.reset();
					tf->netInputW.store(0, std::memory_order_release);
					tf->netInputH.store(0, std::memory_order_release);
				}
			} catch (const std::exception &e) {
				obs_log(LOG_ERROR, "Exception while loading model %s: %s", loadModel.c_str(), e.what());
				std::unique_lock<std::mutex> mlock(tf->modelMutex);
				if (tf->modelEpoch.load(std::memory_order_acquire) == loadEpoch) {
					tf->sessionReady.store(false, std::memory_order_release);
					tf->session.reset();
					tf->netInputW.store(0, std::memory_order_release);
					tf->netInputH.store(0, std::memory_order_release);
				}
			} catch (...) {
				obs_log(LOG_ERROR, "Unknown exception while loading model %s", loadModel.c_str());
				std::unique_lock<std::mutex> mlock(tf->modelMutex);
				if (tf->modelEpoch.load(std::memory_order_acquire) == loadEpoch) {
					tf->sessionReady.store(false, std::memory_order_release);
					tf->session.reset();
					tf->netInputW.store(0, std::memory_order_release);
					tf->netInputH.store(0, std::memory_order_release);
				}
			}

			{
				std::lock_guard<std::mutex> wlock(tf->workerMutex);
				if (!tf->pendingModelLoad &&
				    tf->modelEpoch.load(std::memory_order_acquire) == loadEpoch) {
					tf->modelLoadInFlight.store(false, std::memory_order_release);
				}
			}
			continue;
		}

		const auto t0 = std::chrono::steady_clock::now();

		try {
			std::unique_lock<std::mutex> mlock(tf->modelMutex);
			if (!tf->stopWorker.load(std::memory_order_acquire) &&
			    !tf->isDisabled.load(std::memory_order_acquire) && tf->model && tf->session &&
			    meta.epoch == tf->modelEpoch.load(std::memory_order_acquire)) {
				if (tf->resetRecurrentPending.exchange(false, std::memory_order_acq_rel)) {
					tf->model->resetRecurrentStates(tf->inputTensorValues);
					tf->lastBackgroundMask.release();
					tf->lastImageBGRA.release();
				}

				const cv::Mat &imageInput = tf->cpuInputSlots[static_cast<size_t>(slot)];
				bool skipBySimilarity = false;
				if (params.enableImageSimilarity && !tf->lastImageBGRA.empty() && !imageInput.empty() &&
				    tf->lastImageBGRA.size() == imageInput.size()) {
					if (cv::PSNR(tf->lastImageBGRA, imageInput) > params.imageSimilarityThreshold) {
						skipBySimilarity = true;
					}
				}
				if (!skipBySimilarity && !imageInput.empty()) {
					if (params.enableImageSimilarity) {
						imageInput.copyTo(tf->lastImageBGRA);
					}
					cv::Mat mask;
					processImageForBackground(tf, imageInput, !meta.isBgra, params, mask);
					if (!mask.empty()) {
						if (params.temporalSmoothFactor > 0.0f &&
						    params.temporalSmoothFactor < 1.0f &&
						    !tf->lastBackgroundMask.empty() &&
						    tf->lastBackgroundMask.size() == mask.size()) {
							float effSmooth = params.temporalSmoothFactor;
							if (params.enableThreshold) {
								effSmooth = std::max(effSmooth, params.threshold);
							}
							cv::addWeighted(mask, effSmooth, tf->lastBackgroundMask,
									1.0f - effSmooth, 0.0, mask);
						}
						mask.copyTo(tf->lastBackgroundMask);

						if (params.enableThreshold) {
							applyThresholdMorphology(mask, params, meta.sourceW,
										 meta.sourceH);
						}

						{
							std::lock_guard<std::mutex> outLock(tf->outputLock);
							mask.copyTo(tf->backgroundMask);
							tf->publishedMaskW =
								static_cast<uint32_t>(tf->backgroundMask.cols);
							tf->publishedMaskH =
								static_cast<uint32_t>(tf->backgroundMask.rows);
							tf->publishedMaskGen++;
						}

						const double elapsedMs = std::chrono::duration<double, std::milli>(
										 std::chrono::steady_clock::now() - t0)
										 .count();
						tf->statsFrames++;
						tf->statsAccumMs += elapsedMs;
						if (!tf->loggedFirstFramePath) {
							tf->loggedFirstFramePath = true;
							obs_log(LOG_INFO,
								"First frame completed: path=staged-downscaled+cpu infer_ms=%.2f net=%ux%u",
								elapsedMs, meta.netW, meta.netH);
						} else if (tf->statsFrames % 300 == 0) {
							obs_log(LOG_INFO,
								"Worker stats: path=staged-downscaled+cpu avg_infer_ms=%.2f frames=%llu dropped_by_latest=%llu",
								tf->statsAccumMs / 300.0,
								static_cast<unsigned long long>(tf->statsFrames),
								static_cast<unsigned long long>(droppedSnapshot));
							tf->statsAccumMs = 0.0;
						}
					}
				}
			}
		} catch (const Ort::Exception &e) {
			obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "%s", e.what());
		}

		{
			std::lock_guard<std::mutex> wlock(tf->workerMutex);
			tf->busySlot = -1;
		}
	}
}

void background_filter_video_tick(void *data, float seconds)
{
	// Inference runs asynchronously on workerThread so OBS's tick/render loop never stalls.
	UNUSED_PARAMETER(data);
	UNUSED_PARAMETER(seconds);
}

static bool renderTargetIntoTexrender(obs_source_t *target, gs_texrender_t *texrender, uint32_t dstW, uint32_t dstH)
{
	const uint32_t srcW = obs_source_get_base_width(target);
	const uint32_t srcH = obs_source_get_base_height(target);
	if (srcW == 0 || srcH == 0 || dstW == 0 || dstH == 0) {
		return false;
	}
	gs_texrender_reset(texrender);
	if (!gs_texrender_begin(texrender, dstW, dstH)) {
		return false;
	}
	struct vec4 clearColor;
	vec4_zero(&clearColor);
	gs_clear(GS_CLEAR_COLOR, &clearColor, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(srcW), 0.0f, static_cast<float>(srcH), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(texrender);
	return true;
}

static void captureAndEnqueueLatestFrame(background_removal_filter *tf, obs_source_t *target, uint32_t sourceW,
					 uint32_t sourceH)
{
	const uint32_t netW = tf->netInputW.load(std::memory_order_acquire);
	const uint32_t netH = tf->netInputH.load(std::memory_order_acquire);
	if (netW == 0 || netH == 0) {
		return;
	}

	const int everyX = std::max(1, tf->maskEveryXFrames.load(std::memory_order_relaxed));
	const uint64_t frameIdx = tf->renderFrameCounter++;
	if ((frameIdx % static_cast<uint64_t>(everyX)) != 0) {
		return;
	}

	if (tf->inferTexrender) {
		gs_texture_t *prevTex = gs_texrender_get_texture(tf->inferTexrender);
		if (prevTex && (gs_texture_get_width(prevTex) != netW || gs_texture_get_height(prevTex) != netH)) {
			gs_texrender_destroy(tf->inferTexrender);
			tf->inferTexrender = nullptr;
		}
	}
	if (!tf->inferTexrender) {
		tf->inferTexrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
		if (!tf->inferTexrender) {
			return;
		}
	}
	if (!renderTargetIntoTexrender(target, tf->inferTexrender, netW, netH)) {
		return;
	}
	gs_texture_t *inferTex = gs_texrender_get_texture(tf->inferTexrender);
	if (!inferTex) {
		return;
	}

	int writeSlot = -1;
	uint64_t currentEpoch = 0;
	{
		std::lock_guard<std::mutex> wlock(tf->workerMutex);
		for (int s = 0; s < kInputSlotCount; s++) {
			if (s != tf->busySlot && s != tf->latestSlot && s != tf->renderSlot) {
				writeSlot = s;
				break;
			}
		}
		if (writeSlot == -1) {
			return;
		}
		tf->renderSlot = writeSlot;
		currentEpoch = tf->modelEpoch.load(std::memory_order_acquire);
	}

	bool capturedCpu = false;
	bool capturedIsBgra = false;

	const enum gs_color_format texFmt = gs_texture_get_color_format(inferTex);
	const uint32_t texW = gs_texture_get_width(inferTex);
	const uint32_t texH = gs_texture_get_height(inferTex);
	if ((texFmt == GS_RGBA || texFmt == GS_BGRA) && texW == netW && texH == netH) {
		if (tf->inferStagesurface) {
			const uint32_t sw = gs_stagesurface_get_width(tf->inferStagesurface);
			const uint32_t sh = gs_stagesurface_get_height(tf->inferStagesurface);
			const enum gs_color_format sfmt = gs_stagesurface_get_color_format(tf->inferStagesurface);
			if (sw != netW || sh != netH || sfmt != texFmt) {
				gs_stagesurface_destroy(tf->inferStagesurface);
				tf->inferStagesurface = nullptr;
			}
		}
		if (!tf->inferStagesurface) {
			tf->inferStagesurface = gs_stagesurface_create(netW, netH, texFmt);
		}
		if (tf->inferStagesurface && gs_stagesurface_get_width(tf->inferStagesurface) == texW &&
		    gs_stagesurface_get_height(tf->inferStagesurface) == texH &&
		    gs_stagesurface_get_color_format(tf->inferStagesurface) == texFmt) {
			gs_stage_texture(tf->inferStagesurface, inferTex);
			uint8_t *videoData = nullptr;
			uint32_t linesize = 0;
			if (gs_stagesurface_map(tf->inferStagesurface, &videoData, &linesize)) {
				cv::Mat wrapped(static_cast<int>(netH), static_cast<int>(netW), CV_8UC4, videoData,
						linesize);
				wrapped.copyTo(tf->cpuInputSlots[static_cast<size_t>(writeSlot)]);
				gs_stagesurface_unmap(tf->inferStagesurface);
				capturedCpu = true;
				capturedIsBgra = (texFmt == GS_BGRA);
			}
		}
	}

	{
		std::lock_guard<std::mutex> wlock(tf->workerMutex);
		tf->renderSlot = -1;
		if (capturedCpu && currentEpoch == tf->modelEpoch.load(std::memory_order_acquire) &&
		    !tf->stopWorker.load(std::memory_order_acquire)) {
			if (tf->latestSlot != -1) {
				tf->droppedByLatest++;
			}
			FrameSlotMeta &meta = tf->slotMeta[static_cast<size_t>(writeSlot)];
			meta.isBgra = capturedIsBgra;
			meta.sourceW = sourceW;
			meta.sourceH = sourceH;
			meta.netW = netW;
			meta.netH = netH;
			meta.epoch = currentEpoch;
			tf->latestSlot = writeSlot;
			tf->workerCv.notify_one();
		}
	}
}

static gs_texture_t *blur_background(std::shared_ptr<background_removal_filter> tf, obs_source_t *target,
				     uint32_t width, uint32_t height, gs_texture_t *alphaTexture)
{
	if (tf->blurBackground == 0 || !tf->kawaseBlurEffect) {
		return nullptr;
	}
	gs_texture_t *blurredTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	if (!renderTargetIntoTexrender(target, tf->texrender, width, height)) {
		gs_texture_destroy(blurredTexture);
		return nullptr;
	}
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));

	gs_eparam_t *image = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "image");
	gs_eparam_t *focalmask = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "focalmask");
	gs_eparam_t *xOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "xOffset");
	gs_eparam_t *yOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "yOffset");
	gs_eparam_t *blurIter = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurIter");
	gs_eparam_t *blurTotal = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurTotal");
	gs_eparam_t *blurFocusPointParam = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurFocusPoint");
	gs_eparam_t *blurFocusDepthParam = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurFocusDepth");

	for (int i = 0; i < (int)tf->blurBackground; i++) {
		gs_texrender_reset(tf->texrender);
		if (!gs_texrender_begin(tf->texrender, width, height)) {
			obs_log(LOG_INFO, "Could not open background blur texrender!");
			return blurredTexture;
		}

		gs_effect_set_texture(image, blurredTexture);
		gs_effect_set_texture(focalmask, alphaTexture);
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);
		gs_effect_set_int(blurIter, i);
		gs_effect_set_int(blurTotal, (int)tf->blurBackground);
		gs_effect_set_float(blurFocusPointParam, tf->blurFocusPoint);
		gs_effect_set_float(blurFocusDepthParam, tf->blurFocusDepth);

		struct vec4 background;
		vec4_zero(&background);
		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		const char *blur_type = (tf->enableFocalBlur) ? "DrawFocalBlur" : "Draw";

		while (gs_effect_loop(tf->kawaseBlurEffect, blur_type)) {
			gs_draw_sprite(blurredTexture, 0, width, height);
		}
		gs_blend_state_pop();
		gs_texrender_end(tf->texrender);
		gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	}
	return blurredTexture;
}

static bool ensurePersistentAlphaTexture(background_removal_filter *tf)
{
	std::lock_guard<std::mutex> lock(tf->outputLock);
	if (tf->publishedMaskGen == 0 || tf->backgroundMask.empty()) {
		return false;
	}

	const uint32_t mw = static_cast<uint32_t>(tf->backgroundMask.cols);
	const uint32_t mh = static_cast<uint32_t>(tf->backgroundMask.rows);
	if (!tf->alphaTexture || tf->alphaTexW != mw || tf->alphaTexH != mh) {
		if (tf->alphaTexture) {
			gs_texture_destroy(tf->alphaTexture);
			tf->alphaTexture = nullptr;
		}
		tf->alphaTexture = gs_texture_create(mw, mh, GS_R8, 1, nullptr, GS_DYNAMIC);
		if (!tf->alphaTexture) {
			obs_log(LOG_ERROR, "Failed to create persistent GS_DYNAMIC alpha texture");
			return false;
		}
		tf->alphaTexW = mw;
		tf->alphaTexH = mh;
		tf->uploadedMaskGen = 0;
	}
	if (tf->uploadedMaskGen != tf->publishedMaskGen) {
		gs_texture_set_image(tf->alphaTexture, tf->backgroundMask.data,
				     static_cast<uint32_t>(tf->backgroundMask.step[0]), false);
		tf->uploadedMaskGen = tf->publishedMaskGen;
	}
	return true;
}

void background_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	auto *ptr = static_cast<std::shared_ptr<background_removal_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<background_removal_filter> tf = *ptr;

	if (!tf || tf->isDisabled.load(std::memory_order_acquire) || !obs_source_enabled(tf->source)) {
		if (tf && tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target || !tf->effect) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	const uint32_t width = obs_source_get_base_width(target);
	const uint32_t height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// Capture latest downscaled frame to staged ring buffer & wake worker without blocking render.
	captureAndEnqueueLatestFrame(tf.get(), target, width, height);

	// Update persistent alphaTexture only when a newer mask has been published by the worker.
	if (!ensurePersistentAlphaTexture(tf.get())) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	gs_texture_t *blurredTexture = blur_background(tf, target, width, height, tf->alphaTexture);

	if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		if (blurredTexture) {
			gs_texture_destroy(blurredTexture);
		}
		return;
	}

	gs_eparam_t *alphamask = gs_effect_get_param_by_name(tf->effect, "alphamask");
	gs_eparam_t *blurredBackground = gs_effect_get_param_by_name(tf->effect, "blurredBackground");

	gs_effect_set_texture(alphamask, tf->alphaTexture);

	if (tf->blurBackground > 0 && blurredTexture) {
		gs_effect_set_texture(blurredBackground, blurredTexture);
	}

	gs_blend_state_push();
	gs_reset_blend_state();

	const char *techName;
	if (tf->blurBackground > 0 && blurredTexture) {
		if (tf->enableFocalBlur)
			techName = "DrawWithFocalBlur";
		else
			techName = "DrawWithBlur";
	} else {
		techName = "DrawWithoutBlur";
	}

	obs_source_process_filter_tech_end(tf->source, tf->effect, 0, 0, techName);

	gs_blend_state_pop();

	if (blurredTexture) {
		gs_texture_destroy(blurredTexture);
	}
}
