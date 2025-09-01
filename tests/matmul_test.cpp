#include <gtest/gtest.h>
#include "tensor.h"
#include "tensor_ops.h"

using namespace mininn;

TEST(MatmulTest, BasicMatrixMultiplication)
{
    // test 2x3 * 3x2 = 2x2
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f});
    
    Tensor b({3, 2}, {7.0f, 8.0f,
                      9.0f, 10.0f,
                      11.0f, 12.0f});
    
    Tensor result({2, 2});
    
    TensorOps::matmul(a, b, result);
    
    // expected result:
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    
    EXPECT_FLOAT_EQ(result.at({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(result.at({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(result.at({1, 1}), 154.0f);
}

TEST(MatmulTest, IdentityMatrix)
{
    // test multiplication with identity matrix
    Tensor a({2, 2}, {1.0f, 2.0f,
                      3.0f, 4.0f});
    
    Tensor identity({2, 2}, {1.0f, 0.0f,
                            0.0f, 1.0f});
    
    Tensor result({2, 2});
    
    TensorOps::matmul(a, identity, result);
    
    // should return original matrix
    EXPECT_FLOAT_EQ(result.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(result.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(result.at({1, 1}), 4.0f);
}

TEST(MatmulTest, VectorMultiplication)
{
    // test 1x3 * 3x1 = 1x1 (dot product)
    Tensor a({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor b({3, 1}, {4.0f, 5.0f, 6.0f});
    Tensor result({1, 1});
    
    TensorOps::matmul(a, b, result);
    
    // expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(result.at({0, 0}), 32.0f);
}

TEST(MatmulTest, NonSquareMatrices)
{
    // test 3x2 * 2x4 = 3x4
    Tensor a({3, 2}, {1.0f, 2.0f,
                      3.0f, 4.0f,
                      5.0f, 6.0f});
    
    Tensor b({2, 4}, {1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 1.0f});
    
    Tensor result({3, 4});
    
    TensorOps::matmul(a, b, result);
    
    // verify a few key results
    EXPECT_FLOAT_EQ(result.at({0, 0}), 1.0f);  // 1*1 + 2*0
    EXPECT_FLOAT_EQ(result.at({0, 1}), 2.0f);  // 1*0 + 2*1
    EXPECT_FLOAT_EQ(result.at({2, 0}), 5.0f);  // 5*1 + 6*0
    EXPECT_FLOAT_EQ(result.at({2, 1}), 6.0f);  // 5*0 + 6*1
}

// error condition tests
TEST(MatmulTest, DimensionMismatchError)
{
    Tensor a({2, 3});
    Tensor b({2, 2});  // wrong inner dimension
    Tensor result({2, 2});
    
    EXPECT_THROW(TensorOps::matmul(a, b, result), std::invalid_argument);
}

TEST(MatmulTest, NonTwoDimensionalTensorError)
{
    Tensor a({2, 3, 4});  // 3d tensor
    Tensor b({2, 2});
    Tensor result({2, 2});
    
    EXPECT_THROW(TensorOps::matmul(a, b, result), std::invalid_argument);
}

TEST(MatmulTest, ZeroValues)
{
    // test with matrices containing zeros
    Tensor a({2, 2}, {0.0f, 1.0f,
                      2.0f, 0.0f});
    
    Tensor b({2, 2}, {1.0f, 0.0f,
                      0.0f, 3.0f});
    
    Tensor result({2, 2});
    
    TensorOps::matmul(a, b, result);
    
    EXPECT_FLOAT_EQ(result.at({0, 0}), 0.0f);  // 0*1 + 1*0
    EXPECT_FLOAT_EQ(result.at({0, 1}), 3.0f);  // 0*0 + 1*3
    EXPECT_FLOAT_EQ(result.at({1, 0}), 2.0f);  // 2*1 + 0*0
    EXPECT_FLOAT_EQ(result.at({1, 1}), 0.0f);  // 2*0 + 0*3
}