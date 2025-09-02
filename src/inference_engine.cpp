/* inference_engine.cpp
 * 
 * Implementation of the InferenceEngine class for executing neural network
 * forward passes with performance monitoring and comprehensive error handling.
 */

#include "inference_engine.h"
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace mininn
{
    InferenceEngine::InferenceEngine(std::unique_ptr<Model> model)
        : model_(std::move(model)), profiling_enabled_(false), buffers_allocated_(false)
    {
        if (!model_)
        {
            throw std::invalid_argument("Cannot create inference engine with null model");
        }
        
        if (model_->getLayers().empty())
        {
            throw std::invalid_argument("Cannot create inference engine with empty model");
        }
        
        // validate model has proper input/output shapes
        if (model_->getInputShape().empty() || model_->getOutputShape().empty())
        {
            throw std::invalid_argument("Model must have defined input and output shapes");
        }
        
        // initialize stats
        last_stats_ = InferenceStats{};
    }

    Tensor InferenceEngine::predict(const Tensor& input)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // reset profiling stats
        if (profiling_enabled_)
        {
            last_stats_ = InferenceStats{};
            last_stats_.layer_times.resize(model_->getLayers().size());
        }
        
        // validate input
        validateInput(input);
        
        // execute forward pass
        Tensor output;
        executeForwardPass(input, output);
        
        // update profiling information
        if (profiling_enabled_)
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            last_stats_.total_time = end_time - start_time;
            updateMemoryUsage();
        }
        
        return output;
    }

    std::vector<Tensor> InferenceEngine::predictBatch(const std::vector<Tensor>& inputs)
    {
        if (inputs.empty())
        {
            throw std::invalid_argument("Cannot process empty batch");
        }
        
        std::vector<Tensor> outputs;
        outputs.reserve(inputs.size());
        
        // process each input individually
        // TODO: future optimization -> batch processing through layers
        for (const auto& input : inputs)
        {
            outputs.push_back(predict(input));
        }
        
        return outputs;
    }

    void InferenceEngine::preallocateBuffers()
    {
        if (buffers_allocated_)
        {
            return;
        }
        
        // pre-allocate intermediate tensors for each layer
        // this avoids memory allocation during inference for better performance
        const auto& layers = model_->getLayers();
        intermediate_tensors_.clear();
        intermediate_tensors_.reserve(layers.size() + 1);  // +1 for final output
        
        // start with input shape
        std::vector<size_t> current_shape = model_->getInputShape();
        
        for (size_t i = 0; i < layers.size(); ++i)
        {
            // for now well determine shapes dynamically during first inference
            // more sophisticated engine would infer shapes ahead of time
            intermediate_tensors_.emplace_back();
        }
        
        buffers_allocated_ = true;
    }

    void InferenceEngine::clearBuffers()
    {
        intermediate_tensors_.clear();
        buffers_allocated_ = false;
    }

    void InferenceEngine::validateInput(const Tensor& input) const
    {
        const auto& expected_shape = model_->getInputShape();

        if (input.rank() != expected_shape.size())
        {
            throw std::invalid_argument(
                "Input tensor rank mismatch. Expected: " + std::to_string(expected_shape.size()) +
                ", Got: " + std::to_string(input.rank())
            );
        }
        
        const auto& input_shape = input.shape();
        for (size_t i = 0; i < expected_shape.size(); ++i)
        {
            if (input_shape[i] != expected_shape[i])
            {
                throw std::invalid_argument(
                    "Input tensor shape mismatch at dimension " + std::to_string(i) +
                    ". Expected: " + std::to_string(expected_shape[i]) +
                    ", Got: " + std::to_string(input_shape[i])
                );
            }
        }

        if (input.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument("Input tensor must be FLOAT32 type");
        }
    }

    void InferenceEngine::executeForwardPass(const Tensor& input, Tensor& output)
    {
        const auto& layers = model_->getLayers();
        
        // start with input tensor
        Tensor current_input = input;  // copy input
        Tensor layer_output;
        
        for (size_t i = 0; i < layers.size(); ++i)
        {
            auto layer_start = std::chrono::high_resolution_clock::now();
            
            try 
            {
                // execute layer forward pass
                layers[i]->forward(current_input, layer_output);
                
                // update profiling
                if (profiling_enabled_)
                {
                    auto layer_end = std::chrono::high_resolution_clock::now();
                    last_stats_.layer_times[i] = layer_end - layer_start;
                }
                
                // prepare for next layer (move output to input for next iteration)
                current_input = std::move(layer_output);
                layer_output = Tensor();  // clear for next iteration
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(
                    "Error in layer " + std::to_string(i) + " (type: " + 
                    std::to_string(static_cast<int>(layers[i]->getType())) + "): " + e.what()
                );
            }
        }
        
        // final output
        output = std::move(current_input);
        
        // validate output shape
        const auto& expected_output_shape = model_->getOutputShape();
        if (output.shape() != expected_output_shape)
        {
            std::string expected_str = "[";
            std::string actual_str = "[";
            for (size_t i = 0; i < expected_output_shape.size(); ++i)
            {
                expected_str += std::to_string(expected_output_shape[i]);
                if (i < expected_output_shape.size() - 1) expected_str += ", ";
            }
            for (size_t i = 0; i < output.shape().size(); ++i)
            {
                actual_str += std::to_string(output.shape()[i]);
                if (i < output.shape().size() - 1) actual_str += ", ";
            }
            expected_str += "]";
            actual_str += "]";
            
            throw std::runtime_error(
                "Output shape mismatch. Expected: " + expected_str + ", Got: " + actual_str
            );
        }
    }

    void InferenceEngine::updateMemoryUsage()
    {
        // estimate memory usage (simplified calculation)
        size_t total_bytes = 0;
        
        // count model parameters
        for (const auto& layer : model_->getLayers())
        {
            if (layer->getType() == LayerType::LINEAR)
            {
                // rough estimate for linear layers (weights + bias)
                // this is a simplification - real implementation would track actual sizes
                total_bytes += 1000000;  // placeholder
            }
        }
        
        // add intermediate tensors
        for (const auto& tensor : intermediate_tensors_)
        {
            total_bytes += tensor.size() * sizeof(float);
        }
        
        last_stats_.memory_usage_bytes = total_bytes;
    }

    // factory function
    std::unique_ptr<InferenceEngine> createInferenceEngine(const std::string& model_path)
    {
        try
        {
            auto model = ModelLoader::loadFromFile(model_path);
            return std::make_unique<InferenceEngine>(std::move(model));
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("Failed to create inference engine: " + std::string(e.what()));
        }
    }

    // utility func implementations
    namespace InferenceUtils
    {
        Tensor normalizeInput(const Tensor& input, float mean, float std)
        {
            if (std == 0.0f)
            {
                throw std::invalid_argument("Standard deviation cannot be zero for normalization");
            }
            
            Tensor normalized = input;  // copy
            float* data = normalized.data();
            
            for (size_t i = 0; i < normalized.size(); ++i)
            {
                data[i] = (data[i] - mean) / std;
            }
            
            return normalized;
        }

        Tensor preprocessImage(const std::vector<float>& pixel_data, 
                              size_t width, size_t height, size_t channels)
        {
            if (pixel_data.size() != width * height * channels)
            {
                throw std::invalid_argument("Pixel data size doesn't match dimensions");
            }
            
            std::vector<size_t> shape = {height, width, channels};
            return Tensor(shape, pixel_data);
        }

        std::vector<std::pair<size_t, float>> getTopK(const Tensor& output, size_t k)
        {
            if (output.rank() != 1)
            {
                throw std::invalid_argument("getTopK requires 1D tensor");
            }
            
            if (k > output.size())
            {
                k = output.size();
            }
            
            std::vector<std::pair<size_t, float>> indexed_values;
            indexed_values.reserve(output.size());
            
            const float* data = output.data();
            for (size_t i = 0; i < output.size(); ++i)
            {
                indexed_values.emplace_back(i, data[i]);
            }
            
            // sort by value (descending)
            std::partial_sort(indexed_values.begin(), indexed_values.begin() + k, indexed_values.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            
            indexed_values.resize(k);
            return indexed_values;
        }

        size_t getArgMax(const Tensor& output)
        {
            if (output.rank() != 1)
            {
                throw std::invalid_argument("getArgMax requires 1D tensor");
            }
            
            if (output.size() == 0)
            {
                throw std::invalid_argument("Cannot find argmax of empty tensor");
            }
            
            const float* data = output.data();
            size_t max_idx = 0;
            float max_val = data[0];
            
            for (size_t i = 1; i < output.size(); ++i)
            {
                if (data[i] > max_val)
                {
                    max_val = data[i];
                    max_idx = i;
                }
            }
            
            return max_idx;
        }

        bool isValidModelFile(const std::string& filepath)
        {
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open())
            {
                return false;
            }
            
                // check magic number
            uint32_t magic;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            
            return file.good() && (magic == ModelFormat::MAGIC_NUMBER);
        }

        void validateTensorShape(const Tensor& tensor, const std::vector<size_t>& expected_shape)
        {
            if (tensor.shape() != expected_shape)
            {
                std::string expected_str = "[";
                std::string actual_str = "[";
                
                for (size_t i = 0; i < expected_shape.size(); ++i)
                {
                    expected_str += std::to_string(expected_shape[i]);
                    if (i < expected_shape.size() - 1) expected_str += ", ";
                }
                for (size_t i = 0; i < tensor.shape().size(); ++i)
                {
                    actual_str += std::to_string(tensor.shape()[i]);
                    if (i < tensor.shape().size() - 1) actual_str += ", ";
                }
                expected_str += "]";
                actual_str += "]";
                
                throw std::invalid_argument(
                    "Tensor shape validation failed. Expected: " + expected_str + ", Got: " + actual_str
                );
            }
        }
    }

} // namespace mininn
