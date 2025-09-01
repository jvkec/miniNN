#include <gtest/gtest.h>
#include "tensor.h"
#include "tensor_ops.h"

using namespace mininn;

TEST(ReluTest, PositiveValues)
{
    // relu should leave positive values unchanged
    Tensor tensor({3}, {1.0f, 2.5f, 10.0f});
    
    TensorOps::relu(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor.at({1}), 2.5f);
    EXPECT_FLOAT_EQ(tensor.at({2}), 10.0f);
}

TEST(ReluTest, NegativeValues)
{
    // relu should set negative values to 0
    Tensor tensor({3}, {-1.0f, -2.5f, -10.0f});
    
    TensorOps::relu(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(tensor.at({1}), 0.0f);
    EXPECT_FLOAT_EQ(tensor.at({2}), 0.0f);
}

TEST(ReluTest, ZeroValues)
{
    // relu should leave zero values as zero
    Tensor tensor({3}, {0.0f, 0.0f, 0.0f});
    
    TensorOps::relu(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.0f);
    EXPECT_FLOAT_EQ(tensor.at({1}), 0.0f);
    EXPECT_FLOAT_EQ(tensor.at({2}), 0.0f);
}

TEST(ReluTest, MixedValues)
{
    // test with both positive and negative values
    Tensor tensor({5}, {-2.0f, -0.1f, 0.0f, 0.1f, 2.0f});
    
    TensorOps::relu(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.0f);  // -2.0 -> 0.0
    EXPECT_FLOAT_EQ(tensor.at({1}), 0.0f);  // -0.1 -> 0.0
    EXPECT_FLOAT_EQ(tensor.at({2}), 0.0f);  // 0.0 -> 0.0
    EXPECT_FLOAT_EQ(tensor.at({3}), 0.1f);  // 0.1 -> 0.1
    EXPECT_FLOAT_EQ(tensor.at({4}), 2.0f);  // 2.0 -> 2.0
}

TEST(ReluTest, MultiDimensionalTensor)
{
    // Test with 2D tensor
    Tensor tensor({2, 3}, {-1.0f, 2.0f, -3.0f,
                           4.0f, -5.0f, 6.0f});
    
    TensorOps::relu(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0, 0}), 0.0f);  // -1.0 -> 0.0
    EXPECT_FLOAT_EQ(tensor.at({0, 1}), 2.0f);  // 2.0 -> 2.0
    EXPECT_FLOAT_EQ(tensor.at({0, 2}), 0.0f);  // -3.0 -> 0.0
    EXPECT_FLOAT_EQ(tensor.at({1, 0}), 4.0f);  // 4.0 -> 4.0
    EXPECT_FLOAT_EQ(tensor.at({1, 1}), 0.0f);  // -5.0 -> 0.0
    EXPECT_FLOAT_EQ(tensor.at({1, 2}), 6.0f);  // 6.0 -> 6.0
}

TEST(ReluTest, VerySmallValues)
{
    // test with very small positive and negative values
    Tensor tensor({4}, {-1e-6f, -1e-10f, 1e-10f, 1e-6f});
    
    TensorOps::relu(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.0f);    // small negative -> 0
    EXPECT_FLOAT_EQ(tensor.at({1}), 0.0f);    // very small negative -> 0
    EXPECT_FLOAT_EQ(tensor.at({2}), 1e-10f);  // very small positive unchanged
    EXPECT_FLOAT_EQ(tensor.at({3}), 1e-6f);   // small positive unchanged
}

TEST(ReluTest, LargeValues)
{
    // test with large values
    Tensor tensor({4}, {-1e6f, -1e10f, 1e10f, 1e6f});
    
    TensorOps::relu(tensor);
    
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.0f);   // large negative -> 0
    EXPECT_FLOAT_EQ(tensor.at({1}), 0.0f);   // very large negative -> 0
    EXPECT_FLOAT_EQ(tensor.at({2}), 1e10f);  // very large positive unchanged
    EXPECT_FLOAT_EQ(tensor.at({3}), 1e6f);   // large positive unchanged
}

TEST(ReluTest, EmptyTensorHandling)
{
    // test with minimum valid tensor (1 element)
    Tensor tensor({1}, {-5.0f});
    
    EXPECT_NO_THROW(TensorOps::relu(tensor));
    EXPECT_FLOAT_EQ(tensor.at({0}), 0.0f);
}