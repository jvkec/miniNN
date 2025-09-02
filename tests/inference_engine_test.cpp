/* inference_engine_test.cpp
 * 
 * Tests for the InferenceEngine class, verifying forward pass execution,
 * error handling, and performance monitoring.
 */

#include <gtest/gtest.h>
#include "inference_engine.h"
#include "model_loader.h"
#include <memory>

using namespace mininn;

class InferenceEngineTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        // Create a simple test model programmatically
        model_ = std::make_unique<Model>();
        
        // Add a linear layer: 2 inputs -> 3 outputs
        auto linear = std::make_unique<LinearLayer>(
            Tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),  // weights
            Tensor({3}, {0.1f, 0.2f, 0.3f})                         // bias
        );
        
        // Add ReLU activation
        auto relu = std::make_unique<ReLULayer>();
        
        model_->addLayer(std::move(linear));
        model_->addLayer(std::move(relu));
        
        // Set input/output shapes
        model_->setInputShape({2});
        model_->setOutputShape({3});
    }
    
    std::unique_ptr<Model> model_;
};

// Test engine creation and basic functionality
TEST_F(InferenceEngineTest, EngineCreation) 
{
    EXPECT_NO_THROW({
        InferenceEngine engine(std::move(model_));
        
        EXPECT_EQ(engine.getInputShape(), std::vector<size_t>({2}));
        EXPECT_EQ(engine.getOutputShape(), std::vector<size_t>({3}));
        EXPECT_EQ(engine.getNumLayers(), 2U);
    });
}

TEST_F(InferenceEngineTest, NullModelRejection) 
{
    EXPECT_THROW(
        InferenceEngine engine(nullptr),
        std::invalid_argument
    );
}

TEST_F(InferenceEngineTest, EmptyModelRejection) 
{
    auto empty_model = std::make_unique<Model>();
    
    EXPECT_THROW(
        InferenceEngine engine(std::move(empty_model)),
        std::invalid_argument
    );
}

// Test basic inference
TEST_F(InferenceEngineTest, BasicInference) 
{
    InferenceEngine engine(std::move(model_));
    
    Tensor input({2}, {1.0f, 2.0f});
    Tensor output = engine.predict(input);
    
    // Verify output shape
    EXPECT_EQ(output.shape(), std::vector<size_t>({3}));
    
    // Expected computation:
    // Linear: [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] + [0.1, 0.2, 0.3] = [9.1, 12.2, 15.3]
    // ReLU: max(0, [9.1, 12.2, 15.3]) = [9.1, 12.2, 15.3]
    EXPECT_NEAR(output.data()[0], 9.1f, 1e-5);
    EXPECT_NEAR(output.data()[1], 12.2f, 1e-5);
    EXPECT_NEAR(output.data()[2], 15.3f, 1e-5);
}

TEST_F(InferenceEngineTest, BatchInference) 
{
    InferenceEngine engine(std::move(model_));
    
    std::vector<Tensor> inputs = {
        Tensor({2}, {1.0f, 2.0f}),
        Tensor({2}, {0.5f, 1.5f}),
        Tensor({2}, {-1.0f, 3.0f})
    };
    
    std::vector<Tensor> outputs = engine.predictBatch(inputs);
    
    EXPECT_EQ(outputs.size(), 3U);
    
    // Check first output (same as BasicInference test)
    EXPECT_NEAR(outputs[0].data()[0], 9.1f, 1e-5);
    EXPECT_NEAR(outputs[0].data()[1], 12.2f, 1e-5);
    EXPECT_NEAR(outputs[0].data()[2], 15.3f, 1e-5);
    
    // Verify all outputs have correct shape
    for (const auto& output : outputs) 
    {
        EXPECT_EQ(output.shape(), std::vector<size_t>({3}));
    }
}

TEST_F(InferenceEngineTest, EmptyBatchRejection) 
{
    InferenceEngine engine(std::move(model_));
    
    std::vector<Tensor> empty_batch;
    EXPECT_THROW(
        engine.predictBatch(empty_batch),
        std::invalid_argument
    );
}

// Test input validation
TEST_F(InferenceEngineTest, WrongInputShape) 
{
    InferenceEngine engine(std::move(model_));
    
    // Wrong number of elements
    Tensor wrong_input({3}, {1.0f, 2.0f, 3.0f});
    
    EXPECT_THROW(
        engine.predict(wrong_input),
        std::invalid_argument
    );
}

TEST_F(InferenceEngineTest, WrongInputRank) 
{
    InferenceEngine engine(std::move(model_));
    
    // Wrong rank (2D instead of 1D)
    Tensor wrong_input({1, 2}, {1.0f, 2.0f});
    
    EXPECT_THROW(
        engine.predict(wrong_input),
        std::invalid_argument
    );
}

// Test profiling functionality
TEST_F(InferenceEngineTest, ProfilingDisabledByDefault) 
{
    InferenceEngine engine(std::move(model_));
    
    Tensor input({2}, {1.0f, 2.0f});
    engine.predict(input);
    
    const auto& stats = engine.getLastInferenceStats();
    EXPECT_EQ(stats.total_time.count(), 0.0);  // Should be zero when disabled
}

TEST_F(InferenceEngineTest, ProfilingEnabled) 
{
    InferenceEngine engine(std::move(model_));
    engine.enableProfiling(true);
    
    Tensor input({2}, {1.0f, 2.0f});
    engine.predict(input);
    
    const auto& stats = engine.getLastInferenceStats();
    EXPECT_GT(stats.total_time.count(), 0.0);  // Should have some time recorded
    EXPECT_EQ(stats.layer_times.size(), 2U);   // Should have timings for both layers
}

// Test buffer management
TEST_F(InferenceEngineTest, BufferManagement) 
{
    InferenceEngine engine(std::move(model_));
    
    // Test preallocation
    EXPECT_NO_THROW(engine.preallocateBuffers());
    
    // Test clearing
    EXPECT_NO_THROW(engine.clearBuffers());
}

// Test model with negative outputs (ReLU should clamp them)
TEST_F(InferenceEngineTest, NegativeOutputHandling) 
{
    // Create model that produces negative values before ReLU
    auto test_model = std::make_unique<Model>();
    
    // Linear layer with negative weights/bias
    auto linear = std::make_unique<LinearLayer>(
        Tensor({2, 2}, {-1.0f, -2.0f, -3.0f, -4.0f}),  // negative weights
        Tensor({2}, {-0.5f, -1.0f})                      // negative bias
    );
    auto relu = std::make_unique<ReLULayer>();
    
    test_model->addLayer(std::move(linear));
    test_model->addLayer(std::move(relu));
    test_model->setInputShape({2});
    test_model->setOutputShape({2});
    
    InferenceEngine engine(std::move(test_model));
    
    Tensor input({2}, {1.0f, 1.0f});
    Tensor output = engine.predict(input);
    
    // All outputs should be clamped to 0 by ReLU
    EXPECT_EQ(output.data()[0], 0.0f);
    EXPECT_EQ(output.data()[1], 0.0f);
}

// Test utility functions
TEST_F(InferenceEngineTest, UtilityFunctions) 
{
    // Test normalization
    Tensor input({3}, {2.0f, 4.0f, 6.0f});
    Tensor normalized = InferenceUtils::normalizeInput(input, 2.0f, 2.0f);
    
    // Expected: (x - 2) / 2 = [0, 1, 2]
    EXPECT_NEAR(normalized.data()[0], 0.0f, 1e-5);
    EXPECT_NEAR(normalized.data()[1], 1.0f, 1e-5);
    EXPECT_NEAR(normalized.data()[2], 2.0f, 1e-5);
    
    // Test argmax
    Tensor output({3}, {0.1f, 0.8f, 0.1f});
    size_t max_idx = InferenceUtils::getArgMax(output);
    EXPECT_EQ(max_idx, 1U);
    
    // Test top-k
    Tensor scores({4}, {0.1f, 0.8f, 0.3f, 0.2f});
    auto top_2 = InferenceUtils::getTopK(scores, 2);
    
    EXPECT_EQ(top_2.size(), 2U);
    EXPECT_EQ(top_2[0].first, 1U);  // Index of highest score
    EXPECT_NEAR(top_2[0].second, 0.8f, 1e-5);
    EXPECT_EQ(top_2[1].first, 2U);  // Index of second highest
    EXPECT_NEAR(top_2[1].second, 0.3f, 1e-5);
}

TEST_F(InferenceEngineTest, UtilityErrorHandling) 
{
    // Test normalization with zero std
    Tensor input({2}, {1.0f, 2.0f});
    EXPECT_THROW(
        InferenceUtils::normalizeInput(input, 0.0f, 0.0f),
        std::invalid_argument
    );
    
    // Test argmax on empty tensor (note: can't create tensor with 0 dimensions)
    // This test is commented out since our tensor implementation doesn't allow 0-sized dimensions
    // Tensor empty({0});
    // EXPECT_THROW(
    //     InferenceUtils::getArgMax(empty),
    //     std::invalid_argument
    // );
    
    // Test getTopK on 2D tensor (should fail)
    Tensor matrix({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_THROW(
        InferenceUtils::getTopK(matrix, 2),
        std::invalid_argument
    );
}
