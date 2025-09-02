#pragma once

#include "model_loader.h"
#include "tensor.h"
#include <memory>
#include <vector>
#include <chrono>

namespace mininn
{
    // profiling info
    struct InferenceStats
    {
        std::chrono::duration<double, std::milli> total_time{0};
        std::vector<std::chrono::duration<double, std::milli>> layer_times;
        size_t memory_usage_bytes{0};
    };

    // main inference engine class
    class InferenceEngine
    {
    public:
        // constructor loads model and prepares for inference
        explicit InferenceEngine(std::unique_ptr<Model> model);
        ~InferenceEngine() = default;
        
        // move semantics only (engines can hold large models)
        InferenceEngine(const InferenceEngine&) = delete;
        InferenceEngine& operator=(const InferenceEngine&) = delete;
        InferenceEngine(InferenceEngine&&) = default;
        InferenceEngine& operator=(InferenceEngine&&) = default;
        
        // main inference method -> executes forward pass
        Tensor predict(const Tensor& input);
        
        // batch inference for multiple inputs
        std::vector<Tensor> predictBatch(const std::vector<Tensor>& inputs);
        
        // model introspection
        const std::vector<size_t>& getInputShape() const { return model_->getInputShape(); }
        const std::vector<size_t>& getOutputShape() const { return model_->getOutputShape(); }
        size_t getNumLayers() const { return model_->getLayers().size(); }
        
        // performance monitoring
        void enableProfiling(bool enable) { profiling_enabled_ = enable; }
        const InferenceStats& getLastInferenceStats() const { return last_stats_; }
        
        // mem management
        void preallocateBuffers();  // pre-allocate intermediate tensors for performance
        void clearBuffers();        // free intermediate tensors to save memory
        
    private:
        std::unique_ptr<Model> model_;
        bool profiling_enabled_;
        InferenceStats last_stats_;
        
        // pre-allocated intermediate tensors for performance
        std::vector<Tensor> intermediate_tensors_;
        bool buffers_allocated_;
        
        // helpers
        void validateInput(const Tensor& input) const;
        void executeForwardPass(const Tensor& input, Tensor& output);
        void updateMemoryUsage();
    };

    // factory function for creating inference engines
    std::unique_ptr<InferenceEngine> createInferenceEngine(const std::string& model_path);

    // utility functions for common inference tasks
    namespace InferenceUtils
    {
        // preprocessing helpers
        Tensor normalizeInput(const Tensor& input, float mean = 0.0f, float std = 1.0f);
        Tensor preprocessImage(const std::vector<float>& pixel_data, 
                              size_t width, size_t height, size_t channels);
        
        // postprocessing helpers  
        std::vector<std::pair<size_t, float>> getTopK(const Tensor& output, size_t k);
        size_t getArgMax(const Tensor& output);
        
        // validation helpers
        bool isValidModelFile(const std::string& filepath);
        void validateTensorShape(const Tensor& tensor, const std::vector<size_t>& expected_shape);
    }

} // namespace mininn
