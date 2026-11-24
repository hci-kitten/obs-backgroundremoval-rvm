// SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MODELRVM_H
#define MODELRVM_H

#include "Model.hpp"

class ModelRVM : public ModelBCHW {
private:
	// hq = RVM ResNet50 at 1280x768 with downsample_ratio 0.25 (internal 320x192, all levels /16 exact);
	// the low ratio also enables RVM's deep guided filter, which refines edges at full resolution.
	bool hq;
	float ratio() const { return hq ? 0.25f : 1.0f; }
	int64_t stateChannels(size_t i) const
	{
		static const int64_t mobilenet[5] = {0, 16, 20, 40, 64};
		static const int64_t resnet50[5] = {0, 16, 32, 64, 128};
		return hq ? resnet50[i] : mobilenet[i];
	}

public:
	ModelRVM(bool hq_ = false) : hq(hq_) {}
	~ModelRVM() {}

	virtual bool hasRecurrentStates() const override { return true; }

	virtual float recurrentDownsampleRatio() const override { return ratio(); }

	virtual void resetRecurrentStates(std::vector<std::vector<float>> &inputTensorValues) override
	{
		for (size_t i = 1; i < 5 && i < inputTensorValues.size(); i++) {
			std::fill(inputTensorValues[i].begin(), inputTensorValues[i].end(), 0.0f);
		}
	}

	virtual void populateInputOutputNames(const std::unique_ptr<Ort::Session> &session,
					      std::vector<Ort::AllocatedStringPtr> &inputNames,
					      std::vector<Ort::AllocatedStringPtr> &outputNames)
	{
		Ort::AllocatorWithDefaultOptions allocator;

		inputNames.clear();
		outputNames.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++) {
			inputNames.push_back(session->GetInputNameAllocated(i, allocator));
		}
		for (size_t i = 1; i < session->GetOutputCount(); i++) {
			outputNames.push_back(session->GetOutputNameAllocated(i, allocator));
		}
	}

	virtual bool populateInputOutputShapes(const std::unique_ptr<Ort::Session> &session,
					       std::vector<std::vector<int64_t>> &inputDims,
					       std::vector<std::vector<int64_t>> &outputDims)
	{
		// Assuming model only has one input and one output image

		inputDims.clear();
		outputDims.clear();

		for (size_t i = 0; i < session->GetInputCount(); i++) {
			// Get input shape
			const Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
			const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
			inputDims.push_back(inputTensorInfo.GetShape());
		}

		for (size_t i = 1; i < session->GetOutputCount(); i++) {
			// Get output shape
			const Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
			const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
			outputDims.push_back(outputTensorInfo.GetShape());
		}

		const int base_width = hq ? 1280 : 320;
		const int base_height = hq ? 768 : 192;
		// recurrent states live at the downsampled (internal) resolution
		const int internal_width = (int)(base_width * ratio());
		const int internal_height = (int)(base_height * ratio());

		inputDims[0][0] = 1;
		inputDims[0][1] = 3;
		inputDims[0][2] = base_height;
		inputDims[0][3] = base_width;
		for (size_t i = 1; i < 5; i++) {
			inputDims[i][0] = 1;
			inputDims[i][1] = stateChannels(i);
			inputDims[i][2] = internal_height / (2 << (i - 1));
			inputDims[i][3] = internal_width / (2 << (i - 1));
		}

		outputDims[0][0] = 1;
		outputDims[0][1] = 1;
		outputDims[0][2] = base_height;
		outputDims[0][3] = base_width;
		for (size_t i = 1; i < 5; i++) {
			outputDims[i][0] = 1;
			outputDims[i][1] = stateChannels(i);
			outputDims[i][2] = internal_height / (2 << (i - 1));
			outputDims[i][3] = internal_width / (2 << (i - 1));
		}
		return true;
	}

	virtual void loadInputToTensor(const cv::Mat &preprocessedImage, uint32_t, uint32_t,
				       std::vector<std::vector<float>> &inputTensorValues)
	{
		inputTensorValues[0].assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
		inputTensorValues[5][0] = ratio();
	}

	virtual void assignOutputToInput(std::vector<std::vector<float>> &outputTensorValues,
					 std::vector<std::vector<float>> &inputTensorValues)
	{
		for (size_t i = 1; i < 5; i++) {
			inputTensorValues[i].assign(outputTensorValues[i].begin(), outputTensorValues[i].end());
		}
	}
};

#endif /* MODELRVM_H */
