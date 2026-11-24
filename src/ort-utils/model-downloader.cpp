// SPDX-FileCopyrightText: 2026 hci-kitten
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "model-downloader.hpp"

#include "../consts.h"
#include "../obs-utils/obs-config-utils.hpp"
#include "../plugin-support.h"

#include <curl/curl.h>
#include <obs-module.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace {

struct DownloadableModelSpec {
	const char *modelSelection;
	const char *defaultUrl;
	const char *envUrlVar;
	uint64_t expectedBytes;
	const char *expectedSha256;
};

constexpr uint64_t kMaxDownloadBytes = 120ULL * 1024ULL * 1024ULL;

const DownloadableModelSpec kDownloadableModels[] = {
	{
		MODEL_RVM_RESNET,
		"https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx",
		"OBS_BGREMOVAL_RVM_HQ_URL",
		107479165ULL,
		"25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c",
	},
};

const DownloadableModelSpec *findDownloadableSpec(const std::string &modelSelection)
{
	for (const auto &spec : kDownloadableModels) {
		if (modelSelection == spec.modelSelection) {
			return &spec;
		}
	}
	return nullptr;
}

class Sha256 {
public:
	Sha256() { reset(); }

	void reset()
	{
		state_ = {
			0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
			0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
		};
		bufferLen_ = 0;
		totalBytes_ = 0;
	}

	void update(const uint8_t *data, size_t len)
	{
		if (len == 0) {
			return;
		}
		totalBytes_ += static_cast<uint64_t>(len);
		size_t offset = 0;
		if (bufferLen_ > 0) {
			const size_t fill = std::min(len, sizeof(buffer_) - bufferLen_);
			std::memcpy(buffer_.data() + bufferLen_, data, fill);
			bufferLen_ += fill;
			offset += fill;
			if (bufferLen_ == sizeof(buffer_)) {
				transform(buffer_.data());
				bufferLen_ = 0;
			}
		}
		while (offset + sizeof(buffer_) <= len) {
			transform(data + offset);
			offset += sizeof(buffer_);
		}
		if (offset < len) {
			bufferLen_ = len - offset;
			std::memcpy(buffer_.data(), data + offset, bufferLen_);
		}
	}

	std::string finalHex()
	{
		const uint64_t bitLen = totalBytes_ * 8ULL;
		buffer_[bufferLen_++] = 0x80U;
		if (bufferLen_ > 56) {
			std::memset(buffer_.data() + bufferLen_, 0, sizeof(buffer_) - bufferLen_);
			transform(buffer_.data());
			bufferLen_ = 0;
		}
		std::memset(buffer_.data() + bufferLen_, 0, 56 - bufferLen_);
		for (int i = 0; i < 8; i++) {
			buffer_[56 + static_cast<size_t>(i)] = static_cast<uint8_t>((bitLen >> (56 - i * 8)) & 0xffU);
		}
		transform(buffer_.data());

		static constexpr char kHex[] = "0123456789abcdef";
		std::string out(64, '0');
		for (size_t i = 0; i < 8; i++) {
			const uint32_t v = state_[i];
			for (size_t j = 0; j < 8; j++) {
				out[i * 8 + j] = kHex[(v >> (28 - j * 4)) & 0xfU];
			}
		}
		return out;
	}

private:
	static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32U - n)); }

	void transform(const uint8_t block[64])
	{
		static constexpr uint32_t kRound[64] = {
			0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
			0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
			0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
			0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
			0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
			0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
			0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
			0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
			0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
			0xc67178f2U,
		};

		uint32_t w[64];
		for (size_t i = 0; i < 16; i++) {
			w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
			       (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
			       (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
			       static_cast<uint32_t>(block[i * 4 + 3]);
		}
		for (size_t i = 16; i < 64; i++) {
			const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
			const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
			w[i] = w[i - 16] + s0 + w[i - 7] + s1;
		}

		uint32_t a = state_[0];
		uint32_t b = state_[1];
		uint32_t c = state_[2];
		uint32_t d = state_[3];
		uint32_t e = state_[4];
		uint32_t f = state_[5];
		uint32_t g = state_[6];
		uint32_t h = state_[7];

		for (size_t i = 0; i < 64; i++) {
			const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
			const uint32_t ch = (e & f) ^ ((~e) & g);
			const uint32_t temp1 = h + S1 + ch + kRound[i] + w[i];
			const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
			const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
			const uint32_t temp2 = S0 + maj;

			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}

		state_[0] += a;
		state_[1] += b;
		state_[2] += c;
		state_[3] += d;
		state_[4] += e;
		state_[5] += f;
		state_[6] += g;
		state_[7] += h;
	}

	std::array<uint32_t, 8> state_{};
	std::array<uint8_t, 64> buffer_{};
	size_t bufferLen_ = 0;
	uint64_t totalBytes_ = 0;
};

bool verifyCachedModelFile(const std::string &path, const DownloadableModelSpec &spec)
{
	std::error_code ec;
	if (!std::filesystem::is_regular_file(path, ec) || ec) {
		return false;
	}
	const uint64_t fileSize = std::filesystem::file_size(path, ec);
	if (ec || fileSize != spec.expectedBytes) {
		return false;
	}

	std::ifstream in(path, std::ios::binary);
	if (!in.is_open()) {
		return false;
	}

	Sha256 sha;
	std::array<uint8_t, 65536> chunk{};
	while (in.good()) {
		in.read(reinterpret_cast<char *>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
		const std::streamsize got = in.gcount();
		if (got > 0) {
			sha.update(chunk.data(), static_cast<size_t>(got));
		}
	}
	if (in.bad()) {
		return false;
	}

	return sha.finalHex() == spec.expectedSha256;
}

std::string resolveDownloadUrl(const DownloadableModelSpec &spec, bool &usedOverride)
{
	usedOverride = false;
	if (spec.envUrlVar) {
		const char *envUrl = std::getenv(spec.envUrlVar);
		if (envUrl && envUrl[0] != '\0') {
			const size_t len = std::strlen(envUrl);
			bool hasControl = false;
			for (size_t i = 0; i < len; i++) {
				const unsigned char ch = static_cast<unsigned char>(envUrl[i]);
				if (ch <= 0x20U || ch == 0x7fU) {
					hasControl = true;
					break;
				}
			}
			if (len > 8 && std::strncmp(envUrl, "https://", 8) == 0 && !hasControl) {
				usedOverride = true;
				return std::string(envUrl);
			}
			obs_log(LOG_WARNING, "Ignoring invalid %s override (must be a valid https:// URL)",
				spec.envUrlVar);
		}
	}
	return std::string(spec.defaultUrl);
}

struct DownloadContext {
	FILE *fp = nullptr;
	Sha256 sha;
	uint64_t bytesWritten = 0;
	bool cancelled = false;
	bool writeFailed = false;
	bool exceededLimit = false;
	const std::function<bool()> *shouldCancel = nullptr;
};

static size_t downloadWriteCallback(void *ptr, size_t size, size_t nmemb, void *userdata)
{
	auto *ctx = static_cast<DownloadContext *>(userdata);
	if (ctx->shouldCancel && (*ctx->shouldCancel)()) {
		ctx->cancelled = true;
		return 0;
	}
	if (size == 0 || nmemb == 0) {
		return 0;
	}
	if (size > kMaxDownloadBytes / nmemb) {
		ctx->exceededLimit = true;
		return 0;
	}
	const size_t total = size * nmemb;
	if (ctx->bytesWritten + static_cast<uint64_t>(total) > kMaxDownloadBytes) {
		ctx->exceededLimit = true;
		return 0;
	}
	if (std::fwrite(ptr, 1, total, ctx->fp) != total) {
		ctx->writeFailed = true;
		return 0;
	}
	ctx->sha.update(static_cast<const uint8_t *>(ptr), total);
	ctx->bytesWritten += static_cast<uint64_t>(total);
	return total;
}

static int downloadXferInfoCallback(void *clientp, curl_off_t, curl_off_t, curl_off_t, curl_off_t)
{
	auto *ctx = static_cast<DownloadContext *>(clientp);
	if (ctx->shouldCancel && (*ctx->shouldCancel)()) {
		ctx->cancelled = true;
		return 1;
	}
	return 0;
}

std::atomic<uint64_t> gDownloadPartSeq{0};

} // namespace

bool isDownloadableModel(const std::string &modelSelection)
{
	return findDownloadableSpec(modelSelection) != nullptr;
}

bool resolveExistingModelPath(const std::string &modelSelection, std::string &resolvedPath)
{
	resolvedPath.clear();

	char *bundledPath = obs_module_file(modelSelection.c_str());
	if (bundledPath != nullptr) {
		std::string candidate(bundledPath);
		bfree(bundledPath);
		std::error_code ec;
		if (std::filesystem::is_regular_file(candidate, ec) && !ec) {
			resolvedPath = std::move(candidate);
			return true;
		}
	}

	const DownloadableModelSpec *spec = findDownloadableSpec(modelSelection);
	if (spec == nullptr) {
		return false;
	}

	const std::string cachePath = getModelCachePath(modelSelection);
	if (cachePath.empty()) {
		return false;
	}

	std::error_code ec;
	if (!std::filesystem::exists(cachePath, ec) || ec) {
		return false;
	}

	if (verifyCachedModelFile(cachePath, *spec)) {
		resolvedPath = cachePath;
		return true;
	}

	obs_log(LOG_WARNING, "Cached model %s failed integrity check; removing corrupt file", cachePath.c_str());
	std::filesystem::remove(cachePath, ec);
	return false;
}

bool ensureModelAvailable(const std::string &modelSelection, std::string &resolvedPath,
			  const std::function<bool()> &shouldCancel)
{
	if (resolveExistingModelPath(modelSelection, resolvedPath)) {
		return true;
	}

	const DownloadableModelSpec *spec = findDownloadableSpec(modelSelection);
	if (spec == nullptr) {
		return false;
	}

	if (shouldCancel && shouldCancel()) {
		return false;
	}

	const std::string cachePath = getModelCachePath(modelSelection);
	if (cachePath.empty()) {
		obs_log(LOG_ERROR, "Unable to determine writable config cache path for model %s",
			modelSelection.c_str());
		return false;
	}

	bool usedOverride = false;
	const std::string url = resolveDownloadUrl(*spec, usedOverride);
	const uint64_t seq = gDownloadPartSeq.fetch_add(1, std::memory_order_relaxed);
	std::ostringstream partName;
	partName << cachePath << ".part." << std::hash<std::thread::id>{}(std::this_thread::get_id()) << "." << seq;
	const std::string partPath = partName.str();

	if (usedOverride && spec->envUrlVar) {
		obs_log(LOG_INFO, "Downloading model %s using %s override", modelSelection.c_str(), spec->envUrlVar);
	} else {
		obs_log(LOG_INFO, "Downloading model %s from default release URL", modelSelection.c_str());
	}

	FILE *fp = std::fopen(partPath.c_str(), "wb");
	if (!fp) {
		obs_log(LOG_ERROR, "Failed to open temporary model file %s for writing", partPath.c_str());
		return false;
	}

	CURL *curl = curl_easy_init();
	if (!curl) {
		std::fclose(fp);
		std::error_code ec;
		std::filesystem::remove(partPath, ec);
		obs_log(LOG_ERROR, "Failed to initialize libcurl for model download");
		return false;
	}

	DownloadContext ctx{};
	ctx.fp = fp;
	ctx.shouldCancel = shouldCancel ? &shouldCancel : nullptr;

	const std::string ua = std::string(PLUGIN_NAME) + "/" + PLUGIN_VERSION;
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_USERAGENT, ua.c_str());
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
#if LIBCURL_VERSION_NUM >= 0x075500
	curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "https");
	curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "https");
#else
	curl_easy_setopt(curl, CURLOPT_PROTOCOLS, static_cast<long>(CURLPROTO_HTTPS));
	curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, static_cast<long>(CURLPROTO_HTTPS));
#endif
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
	curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
	curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, downloadWriteCallback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
	curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
	curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, downloadXferInfoCallback);
	curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);

	const CURLcode code = curl_easy_perform(curl);
	curl_easy_cleanup(curl);

	const int flushRc = std::fflush(fp);
	const int closeRc = std::fclose(fp);
	const bool closeOk = (flushRc == 0) && (closeRc == 0);
	std::error_code ec;

	if (ctx.cancelled) {
		std::filesystem::remove(partPath, ec);
		obs_log(LOG_INFO, "Cancelled in-flight download for model %s", modelSelection.c_str());
		return false;
	}

	if (code != CURLE_OK || !closeOk || ctx.writeFailed || ctx.exceededLimit) {
		std::filesystem::remove(partPath, ec);
		obs_log(LOG_ERROR, "Model download failed for %s (curl=%d: %s, bytes=%llu)", modelSelection.c_str(),
			static_cast<int>(code), curl_easy_strerror(code),
			static_cast<unsigned long long>(ctx.bytesWritten));
		return false;
	}

	const std::string actualSha = ctx.sha.finalHex();
	if (ctx.bytesWritten != spec->expectedBytes || actualSha != spec->expectedSha256) {
		std::filesystem::remove(partPath, ec);
		obs_log(LOG_ERROR,
			"Downloaded model %s failed verification (size=%llu expected=%llu, sha256=%s expected=%s)",
			modelSelection.c_str(), static_cast<unsigned long long>(ctx.bytesWritten),
			static_cast<unsigned long long>(spec->expectedBytes), actualSha.c_str(), spec->expectedSha256);
		return false;
	}

	std::filesystem::rename(partPath, cachePath, ec);
	if (ec) {
		std::filesystem::remove(partPath, ec);
		// Another thread may have concurrently moved a verified file into place.
		if (verifyCachedModelFile(cachePath, *spec)) {
			resolvedPath = cachePath;
			return true;
		}
		obs_log(LOG_ERROR, "Failed to move verified model %s into cache path %s", partPath.c_str(),
			cachePath.c_str());
		return false;
	}

	obs_log(LOG_INFO, "Verified and cached model %s at %s (%llu bytes)", modelSelection.c_str(), cachePath.c_str(),
		static_cast<unsigned long long>(ctx.bytesWritten));
	resolvedPath = cachePath;
	return true;
}
