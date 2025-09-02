#pragma once

#include "tensor.h"
#include <vector>
#include <string>
#include <memory>
#include <fstream>

namespace mininn
{
    // layer types supported by our inference engine
    enum class LayerType : uint8_t 
    {
        LINEAR = 0,
        RELU = 1,
        SIGMOID = 2,
        SOFTMAX = 3
    };

    // base class for neural network layers
    class Layer
    {
    public:
        explicit Layer(LayerType type) : type_(type) {}
        virtual ~Layer() = default; // base class destructor is virtual so that the derived classes can have their own destructors
        
        LayerType getType() const { return type_; }
        
        // pure virtual function -> each layer must implement forward pass
        virtual void forward(const Tensor& input, Tensor& output) = 0;
        
    protected:
        LayerType type_;
    };

    // linear (fully connected) layer implementation
    class LinearLayer : public Layer
    {
    public:
        LinearLayer(const Tensor& weights, const Tensor& bias);
        void forward(const Tensor& input, Tensor& output) override;
        
    private:
        Tensor weights_;  // weight matrix [input_size, output_size]
        Tensor bias_;     // bias vector [output_size]
    };

    // activation layers (stateless)
    class ReLULayer : public Layer
    {
    public:
        ReLULayer() : Layer(LayerType::RELU) {}
        void forward(const Tensor& input, Tensor& output) override;
    };

    class SigmoidLayer : public Layer  
    {
    public:
        SigmoidLayer() : Layer(LayerType::SIGMOID) {}
        void forward(const Tensor& input, Tensor& output) override;
    };

    class SoftmaxLayer : public Layer
    {
    public:
        SoftmaxLayer() : Layer(LayerType::SOFTMAX) {}
        void forward(const Tensor& input, Tensor& output) override;
    };

    // nn model container -> this is the main class that holds the layers and metadata
    class Model
    {
    public:
        Model() = default;
        ~Model() = default;
        
        // move semantics only (models can be large)
        Model(const Model&) = delete;
        Model& operator=(const Model&) = delete;
        Model(Model&&) = default;
        Model& operator=(Model&&) = default;
        
        void addLayer(std::unique_ptr<Layer> layer);
        const std::vector<std::unique_ptr<Layer>>& getLayers() const { return layers_; }
        
        // model metadata
        void setInputShape(const std::vector<size_t>& shape) { input_shape_ = shape; }
        void setOutputShape(const std::vector<size_t>& shape) { output_shape_ = shape; }
        const std::vector<size_t>& getInputShape() const { return input_shape_; }
        const std::vector<size_t>& getOutputShape() const { return output_shape_; }
        
    private:
        std::vector<std::unique_ptr<Layer>> layers_;
        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;
    };

    namespace ModelFormat 
    {
        constexpr uint32_t MAGIC_NUMBER = 0x4E4E494D;  // "MINN" in hex
        constexpr uint16_t VERSION_MAJOR = 1;
        constexpr uint16_t VERSION_MINOR = 0;
        
        // file header structure (total: 16 bytes)
        struct Header 
        {
            uint32_t magic;           // magic number for format validation
            uint16_t version_major;   // major version
            uint16_t version_minor;   // minor version  
            uint32_t num_layers;      // number of layers in the model
            uint32_t reserved;        // reserved for future use
        };
    }

    // model loader with comprehensive error handling
    class ModelLoader
    {
    public:
        static std::unique_ptr<Model> loadFromFile(const std::string& filepath);
        static void saveToFile(const Model& model, const std::string& filepath);
        
    private:
        // loading helpers
        static void validateHeader(const ModelFormat::Header& header);
        static std::unique_ptr<Layer> loadLayer(std::ifstream& file);
        static Tensor loadTensor(std::ifstream& file);
        
        // file I/O helpers with error checking
        template<typename T>
        static void readBinary(std::ifstream& file, T& value);
        
        // write binary data to file
        template<typename T>  
        static void writeBinary(std::ofstream& file, const T& value);
    };

} // namespace mininn
