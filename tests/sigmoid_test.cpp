#include <gtest/gtest.h>
#include "tensor.h"
#include "tensor_ops.h"
#include <cmath>

using namespace mininn;

class SigmoidTest : public ::testing::Test 
{
protected:
    // helper function to compute expected sigmoid value
    float expectedSigmoid(float x) 
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

TEST_F(SigmoidTest, ZeroValue)
{
    // sigmoid of 0 should be 0.5
    Tensor tensor({1}, {0.0f});
    
    TensorOps::sigmoid(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.5f);
}

TEST_F(SigmoidTest, PositiveValues)
{
    // test with positive values
    Tensor tensor({3}, {1.0f, 2.0f, 5.0f});
    
    TensorOps::sigmoid(tensor);
    
    EXPECT_NEAR(tensor.at({0}), expectedSigmoid(1.0f), 1e-6f);
    EXPECT_NEAR(tensor.at({1}), expectedSigmoid(2.0f), 1e-6f);
    EXPECT_NEAR(tensor.at({2}), expectedSigmoid(5.0f), 1e-6f);
    
    // Check that all values are between 0 and 1
    EXPECT_GT(tensor.at({0}), 0.0f);
    EXPECT_LT(tensor.at({0}), 1.0f);
    EXPECT_GT(tensor.at({1}), 0.0f);
    EXPECT_LT(tensor.at({1}), 1.0f);
    EXPECT_GT(tensor.at({2}), 0.0f);
    EXPECT_LT(tensor.at({2}), 1.0f);
}

TEST_F(SigmoidTest, NegativeValues)
{
    // test with negative values
    Tensor tensor({3}, {-1.0f, -2.0f, -5.0f});
    
    TensorOps::sigmoid(tensor);
    
    EXPECT_NEAR(tensor.at({0}), expectedSigmoid(-1.0f), 1e-6f);
    EXPECT_NEAR(tensor.at({1}), expectedSigmoid(-2.0f), 1e-6f);
    EXPECT_NEAR(tensor.at({2}), expectedSigmoid(-5.0f), 1e-6f);
    
    // check that all values are between 0 and 1
    EXPECT_GT(tensor.at({0}), 0.0f);
    EXPECT_LT(tensor.at({0}), 1.0f);
    EXPECT_GT(tensor.at({1}), 0.0f);
    EXPECT_LT(tensor.at({1}), 1.0f);
    EXPECT_GT(tensor.at({2}), 0.0f);
    EXPECT_LT(tensor.at({2}), 1.0f);
}

TEST_F(SigmoidTest, SymmetryProperty)
{
    // test sigmoid symmetry: sigmoid(-x) = 1 - sigmoid(x)
    Tensor pos_tensor({3}, {1.0f, 2.0f, 3.0f});
    Tensor neg_tensor({3}, {-1.0f, -2.0f, -3.0f});
    
    TensorOps::sigmoid(pos_tensor);
    TensorOps::sigmoid(neg_tensor);
    
    EXPECT_NEAR(pos_tensor.at({0}) + neg_tensor.at({0}), 1.0f, 1e-6f);
    EXPECT_NEAR(pos_tensor.at({1}) + neg_tensor.at({1}), 1.0f, 1e-6f);
    EXPECT_NEAR(pos_tensor.at({2}) + neg_tensor.at({2}), 1.0f, 1e-6f);
}

TEST_F(SigmoidTest, LargePositiveValues)
{
    // test with large positive values (should approach 1)
    Tensor tensor({3}, {10.0f, 20.0f, 50.0f});  // reduced from 100 to avoid precision issues
    
    TensorOps::sigmoid(tensor);
    
    // should be very close to 1
    EXPECT_NEAR(tensor.at({0}), 1.0f, 1e-4f);
    EXPECT_NEAR(tensor.at({1}), 1.0f, 1e-8f);
    EXPECT_NEAR(tensor.at({2}), 1.0f, 1e-20f);
    
    // should still be less than 1 (but may be exactly 1.0f due to floating point limits)
    EXPECT_LE(tensor.at({0}), 1.0f);
    EXPECT_LE(tensor.at({1}), 1.0f);
    EXPECT_LE(tensor.at({2}), 1.0f);
}

TEST_F(SigmoidTest, LargeNegativeValues)
{
    // test with large negative values (should approach 0)
    Tensor tensor({3}, {-10.0f, -20.0f, -50.0f});  // reduced from -100 to avoid precision issues
    
    TensorOps::sigmoid(tensor);
    
    // should be very close to 0
    EXPECT_NEAR(tensor.at({0}), 0.0f, 1e-4f);
    EXPECT_NEAR(tensor.at({1}), 0.0f, 1e-8f);
    EXPECT_NEAR(tensor.at({2}), 0.0f, 1e-20f);
    
    // should still be greater than or equal to 0 (may be exactly 0.0f due to floating point limits)
    EXPECT_GE(tensor.at({0}), 0.0f);
    EXPECT_GE(tensor.at({1}), 0.0f);
    EXPECT_GE(tensor.at({2}), 0.0f);
}

TEST_F(SigmoidTest, MixedValues)
{
    // test with a mix of positive, negative, and zero values
    Tensor tensor({5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    
    TensorOps::sigmoid(tensor);
    
    // all values should be between 0 and 1
    for (size_t i = 0; i < 5; ++i) 
    {
        EXPECT_GT(tensor.at({i}), 0.0f);
        EXPECT_LT(tensor.at({i}), 1.0f);
    }
    
    // values should be monotonically increasing
    EXPECT_LT(tensor.at({0}), tensor.at({1}));
    EXPECT_LT(tensor.at({1}), tensor.at({2}));
    EXPECT_LT(tensor.at({2}), tensor.at({3}));
    EXPECT_LT(tensor.at({3}), tensor.at({4}));
    
    // middle value should be 0.5
    EXPECT_FLOAT_EQ(tensor.at({2}), 0.5f);
}

TEST_F(SigmoidTest, MultiDimensionalTensor)
{
    // test with 2D tensor
    Tensor tensor({2, 2}, {-1.0f, 1.0f,
                           0.0f, 2.0f});
    
    TensorOps::sigmoid(tensor);
    
    // check expected values
    EXPECT_NEAR(tensor.at({0, 0}), expectedSigmoid(-1.0f), 1e-6f);
    EXPECT_NEAR(tensor.at({0, 1}), expectedSigmoid(1.0f), 1e-6f);
    EXPECT_FLOAT_EQ(tensor.at({1, 0}), 0.5f);  // sigmoid(0) = 0.5
    EXPECT_NEAR(tensor.at({1, 1}), expectedSigmoid(2.0f), 1e-6f);
}

TEST_F(SigmoidTest, MinimalTensor)
{
    // test with minimal valid tensor (1 element)
    Tensor tensor({1}, {0.0f});
    
    EXPECT_NO_THROW(TensorOps::sigmoid(tensor));
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.5f);
}