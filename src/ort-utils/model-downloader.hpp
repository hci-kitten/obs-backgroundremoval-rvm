// SPDX-FileCopyrightText: 2026 hci-kitten
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODEL_DOWNLOADER_HPP
#define MODEL_DOWNLOADER_HPP

#include <functional>
#include <string>

bool isDownloadableModel(const std::string &modelSelection);

bool resolveExistingModelPath(const std::string &modelSelection, std::string &resolvedPath);

bool ensureModelAvailable(const std::string &modelSelection, std::string &resolvedPath,
			  const std::function<bool()> &shouldCancel = {});

#endif /* MODEL_DOWNLOADER_HPP */
