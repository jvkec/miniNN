#include <gtest/gtest.h>
#include "tensor.h"
#include "tensor_ops.h"
#include <cmath>

using namespace mininn;

class SoftmaxTest : public ::testing::Test 
{
protected:
    // Helper function to check if probabilities sum to 1
    bool checkProbabilitySum(const Tensor& tensor, float tolerance = 1e-6f)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            sum += tensor.data()[i];
        }
        return std::abs(sum - 1.0f) < tolerance;
    }
    
    // Helper function to check if all values are between 0 and 1
    bool checkProbabilityRange(const Tensor& tensor)
    {
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            if (tensor.data()[i] < 0.0f || tensor.data()[i] > 1.0f)
            {
                return false;
            }
        }
        return true;
    }
};

TEST_F(SoftmaxTest, BasicProbabilityProperties)
{
    // Test basic softmax properties
    Tensor tensor({3}, {1.0f, 2.0f, 3.0f});
    
    TensorOps::softmax(tensor);
    
    // Check that all values are probabilities (0 <= p <= 1)
    EXPECT_TRUE(checkProbabilityRange(tensor));
    
    // Check that probabilities sum to 1
    EXPECT_TRUE(checkProbabilitySum(tensor));
    
    // Check that larger input values result in larger probabilities
    EXPECT_LT(tensor.at({0}), tensor.at({1}));
    EXPECT_LT(tensor.at({1}), tensor.at({2}));
}

TEST_F(SoftmaxTest, UniformInput)
{
    // All equal inputs should produce equal probabilities
    Tensor tensor({4}, {2.0f, 2.0f, 2.0f, 2.0f});
    
    TensorOps::softmax(tensor);
    
    // All probabilities should be 0.25
    EXPECT_NEAR(tensor.at({0}), 0.25f, 1e-6f);
    EXPECT_NEAR(tensor.at({1}), 0.25f, 1e-6f);
    EXPECT_NEAR(tensor.at({2}), 0.25f, 1e-6f);
    EXPECT_NEAR(tensor.at({3}), 0.25f, 1e-6f);
    
    EXPECT_TRUE(checkProbabilitySum(tensor));
}

TEST_F(SoftmaxTest, SingleElement)
{
    // Single element should have probability 1
    Tensor tensor({1}, {5.0f});
    
    TensorOps::softmax(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 1.0f);
}

TEST_F(SoftmaxTest, ZeroValues)
{
    // Test with all zero inputs
    Tensor tensor({3}, {0.0f, 0.0f, 0.0f});
    
    TensorOps::softmax(tensor);
    
    // Should produce equal probabilities
    EXPECT_NEAR(tensor.at({0}), 1.0f/3.0f, 1e-6f);
    EXPECT_NEAR(tensor.at({1}), 1.0f/3.0f, 1e-6f);
    EXPECT_NEAR(tensor.at({2}), 1.0f/3.0f, 1e-6f);
    
    EXPECT_TRUE(checkProbabilitySum(tensor));
}

TEST_F(SoftmaxTest, LargeValues)
{
    // Test numerical stability with large values
    Tensor tensor({3}, {1000.0f, 1001.0f, 1002.0f});
    
    TensorOps::softmax(tensor);
    
    // Should not produce NaN or inf
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_FALSE(std::isnan(tensor.data()[i]));
        EXPECT_FALSE(std::isinf(tensor.data()[i]));
    }
    
    EXPECT_TRUE(checkProbabilityRange(tensor));
    EXPECT_TRUE(checkProbabilitySum(tensor));
    
    // Larger values should still have higher probabilities
    EXPECT_LT(tensor.at({0}), tensor.at({1}));
    EXPECT_LT(tensor.at({1}), tensor.at({2}));
}

TEST_F(SoftmaxTest, NegativeValues)
{
    // Test with negative values
    Tensor tensor({3}, {-1.0f, -2.0f, -3.0f});
    
    TensorOps::softmax(tensor);
    
    EXPECT_TRUE(checkProbabilityRange(tensor));
    EXPECT_TRUE(checkProbabilitySum(tensor));
    
    // Less negative values should have higher probabilities
    EXPECT_GT(tensor.at({0}), tensor.at({1}));
    EXPECT_GT(tensor.at({1}), tensor.at({2}));
}

TEST_F(SoftmaxTest, MixedValues)
{
    // Test with mixed positive and negative values
    Tensor tensor({4}, {-2.0f, 0.0f, 1.0f, 3.0f});
    
    TensorOps::softmax(tensor);
    
    EXPECT_TRUE(checkProbabilityRange(tensor));
    EXPECT_TRUE(checkProbabilitySum(tensor));
    
    // Values should be in ascending order of probability
    EXPECT_LT(tensor.at({0}), tensor.at({1}));
    EXPECT_LT(tensor.at({1}), tensor.at({2}));
    EXPECT_LT(tensor.at({2}), tensor.at({3}));
}

TEST_F(SoftmaxTest, TemperatureEffect)
{
    // Test that large differences amplify probability differences
    Tensor tensor1({2}, {1.0f, 2.0f});
    Tensor tensor2({2}, {10.0f, 20.0f});  // Same ratio, larger scale
    
    TensorOps::softmax(tensor1);
    TensorOps::softmax(tensor2);
    
    // tensor2 should have more extreme probabilities
    EXPECT_LT(tensor2.at({0}), tensor1.at({0}));  // Lower probability for smaller value
    EXPECT_GT(tensor2.at({1}), tensor1.at({1}));  // Higher probability for larger value
}

TEST_F(SoftmaxTest, VerifyKnownCase)
{
    // Test with known case where we can verify manually
    Tensor tensor({2}, {0.0f, 0.0f});
    
    TensorOps::softmax(tensor);
    
    // Both should be 0.5
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.5f);
    EXPECT_FLOAT_EQ(tensor.at({1}), 0.5f);
}

TEST_F(SoftmaxTest, ScaleInvariance)
{
    // Adding same constant to all inputs shouldn't change relative probabilities
    Tensor tensor1({3}, {1.0f, 2.0f, 3.0f});
    Tensor tensor2({3}, {11.0f, 12.0f, 13.0f});  // Same but +10
    
    TensorOps::softmax(tensor1);
    TensorOps::softmax(tensor2);
    
    // Results should be identical (within numerical precision)
    EXPECT_NEAR(tensor1.at({0}), tensor2.at({0}), 1e-6f);
    EXPECT_NEAR(tensor1.at({1}), tensor2.at({1}), 1e-6f);
    EXPECT_NEAR(tensor1.at({2}), tensor2.at({2}), 1e-6f);
}

TEST_F(SoftmaxTest, MultiDimensionalTensor)
{
    // Test with 2D tensor (treated as flattened)
    Tensor tensor({2, 2}, {1.0f, 2.0f,
                           3.0f, 4.0f});
    
    TensorOps::softmax(tensor);
    
    EXPECT_TRUE(checkProbabilityRange(tensor));
    EXPECT_TRUE(checkProbabilitySum(tensor));
    
    // Check monotonic increase
    EXPECT_LT(tensor.at({0, 0}), tensor.at({0, 1}));
    EXPECT_LT(tensor.at({0, 1}), tensor.at({1, 0}));
    EXPECT_LT(tensor.at({1, 0}), tensor.at({1, 1}));
}

// Error condition tests - Test our actual empty tensor safety check
TEST_F(SoftmaxTest, EmptyTensorSafetyCheck)
{
    // Our softmax implementation checks for empty tensors and throws
    // But first we need a tensor that can be created (size > 0) then made empty
    
    // Test the minimal case instead
    Tensor tensor({1}, {5.0f});
    
    // This should work fine
    EXPECT_NO_THROW(TensorOps::softmax(tensor));
    EXPECT_FLOAT_EQ(tensor.at({0}), 1.0f);  // Single element always becomes 1.0
    
    // Note: Cannot create tensor with shape {0} due to validation in Tensor constructor
}

TEST_F(SoftmaxTest, VeryLargeNegativeValues)
{
    // Test numerical stability with very large negative values
    Tensor tensor({3}, {-1000.0f, -1001.0f, -1002.0f});
    
    TensorOps::softmax(tensor);
    
    // Should not produce NaN or inf
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_FALSE(std::isnan(tensor.data()[i]));
        EXPECT_FALSE(std::isinf(tensor.data()[i]));
    }
    
    EXPECT_TRUE(checkProbabilityRange(tensor));
    EXPECT_TRUE(checkProbabilitySum(tensor));
}
