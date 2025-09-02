/* model_loader.cpp
 * 
 * Implementation of the ModelLoader class for loading neural network models
 * from binary files.
 */

#include "model_loader.h"
#include "tensor_ops.h"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstring>

namespace mininn
{
    LinearLayer::LinearLayer(const Tensor& weights, const Tensor& bias)
        : Layer(LayerType::LINEAR), weights_(weights), bias_(bias)
    {
        // validate dimensions
        if (weights.rank() != 2)
        {
            throw std::invalid_argument("Linear layer weights must be 2D tensor");
        }
        if (bias.rank() != 1)
        {
            throw std::invalid_argument("Linear layer bias must be 1D tensor");
        }
        if (weights.shape()[1] != bias.shape()[0])
        {
            throw std::invalid_argument(
                "Weight output dimension must match bias dimension: " +
                std::to_string(weights.shape()[1]) + " != " + std::to_string(bias.shape()[0])
            );
        }
    }

    void LinearLayer::forward(const Tensor& input, Tensor& output)
    {
        // Linear transformation: output = input * weights + bias
        // input: [batch_size, input_features] or [input_features]  
        // weights: [input_features, output_features]
        // bias: [output_features]
        // output: [batch_size, output_features] or [output_features]
        
        if (input.rank() == 1)
        {
            // single sample: input [input_features]
            if (input.shape()[0] != weights_.shape()[0])
            {
                throw std::invalid_argument(
                    "Input features must match weight input dimension: " +
                    std::to_string(input.shape()[0]) + " != " + std::to_string(weights_.shape()[0])
                );
            }
            
            // reshape input to [1, input_features] for matrix multiplication
            Tensor input_2d({1, input.shape()[0]});
            std::memcpy(input_2d.data(), input.data(), input.size() * sizeof(float));
            
            // output_temp: [1, output_features]
            Tensor output_temp;
            TensorOps::matmul(input_2d, weights_, output_temp);
            
            // create output tensor and add bias
            output = Tensor({weights_.shape()[1]});
            for (size_t i = 0; i < output.size(); ++i)
            {
                output.data()[i] = output_temp.data()[i] + bias_.data()[i];
            }
        }
        else if (input.rank() == 2)
        {
            // batch processing: input [batch_size, input_features]
            if (input.shape()[1] != weights_.shape()[0])
            {
                throw std::invalid_argument(
                    "Input features must match weight input dimension: " +
                    std::to_string(input.shape()[1]) + " != " + std::to_string(weights_.shape()[0])
                );
            }
            
            // matrix multiplication: [batch_size, input_features] * [input_features, output_features]
            TensorOps::matmul(input, weights_, output);
            
            // add bias to each sample in the batch
            const size_t batch_size = output.shape()[0];
            const size_t output_features = output.shape()[1];
            
            for (size_t batch = 0; batch < batch_size; ++batch)
            {
                for (size_t feature = 0; feature < output_features; ++feature)
                {
                    output.at({batch, feature}) += bias_.at({feature});
                }
            }
        }
        else
        {
            throw std::invalid_argument("Linear layer input must be 1D or 2D tensor");
        }
    }

    // Activation layer implementations
    void ReLULayer::forward(const Tensor& input, Tensor& output)
    {
        // copy input to output then apply ReLU in-place
        output = input;  // this uses the copy constructor
        TensorOps::relu(output);
    }

    void SigmoidLayer::forward(const Tensor& input, Tensor& output)
    {
        output = input;
        TensorOps::sigmoid(output);
    }

    void SoftmaxLayer::forward(const Tensor& input, Tensor& output)
    {
        output = input;
        TensorOps::softmax(output);
    }

    // Model implementation
    void Model::addLayer(std::unique_ptr<Layer> layer)
    {
        if (!layer)
        {
            throw std::invalid_argument("Cannot add null layer to model");
        }
        layers_.push_back(std::move(layer));
    }

    // ModelLoader implementation
    std::unique_ptr<Model> ModelLoader::loadFromFile(const std::string& filepath)
    {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open model file: " + filepath);
        }

        try
        {
            // read + validate header
            ModelFormat::Header header;
            readBinary(file, header);
            validateHeader(header);

            auto model = std::make_unique<Model>();

            // load each layer
            for (uint32_t i = 0; i < header.num_layers; ++i)
            {
                auto layer = loadLayer(file);
                model->addLayer(std::move(layer));
            }

            // load input/output shape metadata
            std::vector<size_t> input_shape, output_shape;
            
            // read input shape
            uint32_t input_rank;
            readBinary(file, input_rank);
            input_shape.resize(input_rank);
            for (uint32_t i = 0; i < input_rank; ++i)
            {
                uint32_t dim;
                readBinary(file, dim);
                input_shape[i] = dim;
            }
            
            // read output shape  
            uint32_t output_rank;
            readBinary(file, output_rank);
            output_shape.resize(output_rank);
            for (uint32_t i = 0; i < output_rank; ++i)
            {
                uint32_t dim;
                readBinary(file, dim);
                output_shape[i] = dim;
            }
            
            model->setInputShape(input_shape);
            model->setOutputShape(output_shape);

            return model;
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("Failed to load model from " + filepath + ": " + e.what());
        }
    }

    void ModelLoader::validateHeader(const ModelFormat::Header& header)
    {
        if (header.magic != ModelFormat::MAGIC_NUMBER)
        {
            throw std::runtime_error("Invalid model file format (magic number mismatch)");
        }
        
        if (header.version_major != ModelFormat::VERSION_MAJOR)
        {
            throw std::runtime_error(
                "Unsupported model version: " + std::to_string(header.version_major) + 
                "." + std::to_string(header.version_minor)
            );
        }
        
        if (header.num_layers == 0)
        {
            throw std::runtime_error("Model must contain at least one layer");
        }
        
        if (header.num_layers > 1000)  // reasonable(?) upper bound
        {
            throw std::runtime_error("Model contains too many layers: " + std::to_string(header.num_layers));
        }
    }

    std::unique_ptr<Layer> ModelLoader::loadLayer(std::ifstream& file)
    {
        uint8_t layer_type_raw;
        readBinary(file, layer_type_raw);
        
        LayerType layer_type = static_cast<LayerType>(layer_type_raw);
        
        switch (layer_type)
        {
            case LayerType::LINEAR:
            {
                Tensor weights = loadTensor(file);
                Tensor bias = loadTensor(file);
                return std::make_unique<LinearLayer>(weights, bias);
            }
            
            case LayerType::RELU:
                return std::make_unique<ReLULayer>();
                
            case LayerType::SIGMOID:
                return std::make_unique<SigmoidLayer>();
                
            case LayerType::SOFTMAX:
                return std::make_unique<SoftmaxLayer>();
                
            default:
                throw std::runtime_error("Unknown layer type: " + std::to_string(layer_type_raw));
        }
    }

    Tensor ModelLoader::loadTensor(std::ifstream& file)
    {
        // Read tensor metadata
        uint8_t dtype_raw;
        readBinary(file, dtype_raw);
        DataType dtype = static_cast<DataType>(dtype_raw);
        
        uint32_t rank;
        readBinary(file, rank);
        
        if (rank == 0 || rank > 8)  // reasonable(?) bounds
        {
            throw std::runtime_error("Invalid tensor rank: " + std::to_string(rank));
        }
        
        std::vector<size_t> shape(rank);
        for (uint32_t i = 0; i < rank; ++i)
        {
            uint32_t dim;
            readBinary(file, dim);
            shape[i] = dim;
        }
        
        // create tensor and read data
        Tensor tensor(shape, dtype);
        
        // for now, we only support FLOAT32 data
        if (dtype != DataType::FLOAT32)
        {
            throw std::runtime_error("Only FLOAT32 tensors are currently supported");
        }
        
        file.read(reinterpret_cast<char*>(tensor.data()), tensor.size() * sizeof(float));
        if (!file.good())
        {
            throw std::runtime_error("Failed to read tensor data");
        }
        
        return tensor;
    }

    // template specializations for binary I/O
    template<typename T>
    void ModelLoader::readBinary(std::ifstream& file, T& value)
    {
        file.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!file.good())
        {
            throw std::runtime_error("Failed to read binary data from file");
        }
    }

    template<typename T>
    void ModelLoader::writeBinary(std::ofstream& file, const T& value)
    {
        file.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!file.good())
        {
            throw std::runtime_error("Failed to write binary data to file");
        }
    }

    // save functionality (for completeness)
    void ModelLoader::saveToFile(const Model& /* model */, const std::string& /* filepath */)
    {
        // TODO: implement model saving functionality
        // this would be useful for creating test models or converting from other formats
        throw std::runtime_error("Model saving not yet implemented");
    }

} // namespace mininn
