/* model_loader_test.cpp
 * 
 * Tests for the ModelLoader class, verifying model loading, validation,
 * and error handling capabilities.
 */

#include <gtest/gtest.h>
#include "model_loader.h"
#include "inference_engine.h"
#include <fstream>
#include <vector>

using namespace mininn;

class ModelLoaderTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        // create a temporary directory for test files
        test_file_path_ = "/tmp/test_model.minn";
    }
    
    void TearDown() override 
    {
        // clean up test files
        std::remove(test_file_path_.c_str());
    }
    
    // helper to create a simple valid model file for testing
    void createSimpleModelFile() 
    {
        std::ofstream file(test_file_path_, std::ios::binary);
        
        // write header
        ModelFormat::Header header;
        header.magic = ModelFormat::MAGIC_NUMBER;
        header.version_major = ModelFormat::VERSION_MAJOR;
        header.version_minor = ModelFormat::VERSION_MINOR;
        header.num_layers = 2;  // linear + ReLU
        header.reserved = 0;
        
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        
        // layer 1: linear layer
        uint8_t layer_type = static_cast<uint8_t>(LayerType::LINEAR);
        file.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));
        
        // weights tensor: 2x3 matrix
        uint8_t dtype = static_cast<uint8_t>(DataType::FLOAT32);
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        uint32_t rank = 2;
        file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        uint32_t dim1 = 2, dim2 = 3;
        file.write(reinterpret_cast<const char*>(&dim1), sizeof(dim1));
        file.write(reinterpret_cast<const char*>(&dim2), sizeof(dim2));
        
        std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
        
        // bias tensor: 3 elements
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        rank = 1;
        file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        uint32_t bias_dim = 3;
        file.write(reinterpret_cast<const char*>(&bias_dim), sizeof(bias_dim));
        
        std::vector<float> bias = {0.1f, 0.2f, 0.3f};
        file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
        
        // layer 2: ReLU layer
        layer_type = static_cast<uint8_t>(LayerType::RELU);
        file.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));
        
        // input shape: [2]
        uint32_t input_rank = 1;
        file.write(reinterpret_cast<const char*>(&input_rank), sizeof(input_rank));
        uint32_t input_dim = 2;
        file.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
        
        // output shape: [3]
        uint32_t output_rank = 1;
        file.write(reinterpret_cast<const char*>(&output_rank), sizeof(output_rank));
        uint32_t output_dim = 3;
        file.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
    }
    
    void createInvalidMagicFile() 
    {
        std::ofstream file(test_file_path_, std::ios::binary);
        uint32_t invalid_magic = 0x12345678;
        file.write(reinterpret_cast<const char*>(&invalid_magic), sizeof(invalid_magic));
    }
    
    std::string test_file_path_;
};

// test individual layer creation
TEST_F(ModelLoaderTest, LinearLayerCreation) 
{
    Tensor weights({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor bias({3}, {0.1f, 0.2f, 0.3f});
    
    EXPECT_NO_THROW({
        LinearLayer layer(weights, bias);
        EXPECT_EQ(layer.getType(), LayerType::LINEAR);
    });
}

TEST_F(ModelLoaderTest, LinearLayerInvalidDimensions) 
{
    Tensor weights({2, 3});  // 2x3
    Tensor bias({2});        // Size 2 (should be 3)
    
    EXPECT_THROW(LinearLayer layer(weights, bias), std::invalid_argument);
}

TEST_F(ModelLoaderTest, ActivationLayers) 
{
    ReLULayer relu;
    SigmoidLayer sigmoid;
    SoftmaxLayer softmax;
    
    EXPECT_EQ(relu.getType(), LayerType::RELU);
    EXPECT_EQ(sigmoid.getType(), LayerType::SIGMOID);
    EXPECT_EQ(softmax.getType(), LayerType::SOFTMAX);
}

// test model creation and manipulation
TEST_F(ModelLoaderTest, ModelCreation) 
{
    Model model;
    
    // add layers
    auto linear = std::make_unique<LinearLayer>(
        Tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
        Tensor({3}, {0.1f, 0.2f, 0.3f})
    );
    auto relu = std::make_unique<ReLULayer>();
    
    model.addLayer(std::move(linear));
    model.addLayer(std::move(relu));
    
    EXPECT_EQ(model.getLayers().size(), 2U);
    
    // set shapes
    model.setInputShape({2});
    model.setOutputShape({3});
    
    EXPECT_EQ(model.getInputShape(), std::vector<size_t>({2}));
    EXPECT_EQ(model.getOutputShape(), std::vector<size_t>({3}));
}

TEST_F(ModelLoaderTest, ModelNullLayerRejection) 
{
    Model model;
    EXPECT_THROW(model.addLayer(nullptr), std::invalid_argument);
}

// test file validation
TEST_F(ModelLoaderTest, NonExistentFile) 
{
    EXPECT_THROW(
        ModelLoader::loadFromFile("/nonexistent/path/model.minn"),
        std::runtime_error
    );
}

TEST_F(ModelLoaderTest, InvalidMagicNumber) 
{
    createInvalidMagicFile();
    
    EXPECT_THROW(
        ModelLoader::loadFromFile(test_file_path_),
        std::runtime_error
    );
}

// test utility functions
TEST_F(ModelLoaderTest, ValidModelFileCheck) 
{
    // non-existent file
    EXPECT_FALSE(InferenceUtils::isValidModelFile("/nonexistent/file.minn"));
    
    // invalid magic number
    createInvalidMagicFile();
    EXPECT_FALSE(InferenceUtils::isValidModelFile(test_file_path_));
    
    // valid file (if we had one)
    // Note: Wed need to create a valid file to test this properly
}

TEST_F(ModelLoaderTest, SaveToFileNotImplemented) 
{
    Model model;
    EXPECT_THROW(
        ModelLoader::saveToFile(model, "test.minn"),
        std::runtime_error
    );
}

// test layer forward pass functionality
TEST_F(ModelLoaderTest, LinearLayerForward) 
{
    // create a simple 2->3 linear layer
    Tensor weights({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor bias({3}, {0.1f, 0.2f, 0.3f});
    LinearLayer layer(weights, bias);
    
    // test with 1D input
    Tensor input({2}, {1.0f, 2.0f});
    Tensor output;
    
    EXPECT_NO_THROW(layer.forward(input, output));
    
    // expected: [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] + [0.1, 0.2, 0.3]
    //         = [9, 12, 15] + [0.1, 0.2, 0.3] = [9.1, 12.2, 15.3]
    EXPECT_EQ(output.shape(), std::vector<size_t>({3}));
    EXPECT_NEAR(output.data()[0], 9.1f, 1e-5);
    EXPECT_NEAR(output.data()[1], 12.2f, 1e-5);
    EXPECT_NEAR(output.data()[2], 15.3f, 1e-5);
}

TEST_F(ModelLoaderTest, ActivationLayerForward) 
{
    // test ReLU
    ReLULayer relu;
    Tensor input({3}, {-1.0f, 0.0f, 2.0f});
    Tensor output;
    
    relu.forward(input, output);
    EXPECT_EQ(output.data()[0], 0.0f);
    EXPECT_EQ(output.data()[1], 0.0f);
    EXPECT_EQ(output.data()[2], 2.0f);
    
    // test Sigmoid
    SigmoidLayer sigmoid;
    Tensor input2({1}, {0.0f});
    Tensor output2;
    
    sigmoid.forward(input2, output2);
    EXPECT_NEAR(output2.data()[0], 0.5f, 1e-5);
}
