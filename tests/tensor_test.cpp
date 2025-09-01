#include <gtest/gtest.h>
#include "tensor.h"
#include <vector>
#include <stdexcept>

using namespace mininn;

class TensorTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        // random test data
        shape2d = {2, 3};
        shape3d = {2, 3, 4};
        data_2x3 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    }

    std::vector<size_t> shape2d;
    std::vector<size_t> shape3d;
    std::vector<float> data_2x3;
};

// testing constructors
TEST_F(TensorTest, DefaultConstructor) 
{
    Tensor t;
    EXPECT_EQ(t.rank(), 0U);
    EXPECT_EQ(t.size(), 0U);
    EXPECT_EQ(t.dtype(), DataType::FLOAT32);
}

TEST_F(TensorTest, ShapeConstructor) 
{
    Tensor t(shape2d);
    EXPECT_EQ(t.rank(), 2U);
    EXPECT_EQ(t.size(), 6U);
    EXPECT_EQ(t.shape(), shape2d);
    EXPECT_EQ(t.dtype(), DataType::FLOAT32);
    EXPECT_NE(t.data(), nullptr);
}

TEST_F(TensorTest, ShapeWithDataConstructor) 
{
    Tensor t(shape2d, data_2x3);
    EXPECT_EQ(t.rank(), 2U);
    EXPECT_EQ(t.size(), 6U);
    EXPECT_EQ(t.shape(), shape2d);
    
    // check data was copied correctly
    for (size_t i = 0; i < data_2x3.size(); ++i) 
    {
        EXPECT_FLOAT_EQ(t.data()[i], data_2x3[i]);
    }
}

TEST_F(TensorTest, ConstructorWithDifferentDataType) 
{
    Tensor t(shape2d, DataType::INT8);
    EXPECT_EQ(t.dtype(), DataType::INT8);
}

// testing validation
TEST_F(TensorTest, InvalidShapeEmpty) 
{
    std::vector<size_t> empty_shape;
    EXPECT_THROW(Tensor t(empty_shape), std::invalid_argument);
}

TEST_F(TensorTest, InvalidShapeZeroDimension) 
{
    std::vector<size_t> invalid_shape = {2, 0, 3};
    EXPECT_THROW(Tensor t(invalid_shape), std::invalid_argument);
}

TEST_F(TensorTest, DataSizeMismatch) 
{
    std::vector<float> wrong_data = {1.0f, 2.0f, 3.0f}; // only 3 elements for 2x3 tensor
    EXPECT_THROW(Tensor t(shape2d, wrong_data), std::invalid_argument);
}

// testing copy operations
TEST_F(TensorTest, CopyConstructor) 
{
    Tensor original(shape2d, data_2x3);
    Tensor copy(original);
    
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.dtype(), original.dtype());
    EXPECT_NE(copy.data(), original.data()); // different memory addresses
    
    // check data is the same
    for (size_t i = 0; i < original.size(); ++i) 
    {
        EXPECT_FLOAT_EQ(copy.data()[i], original.data()[i]);
    }
}

TEST_F(TensorTest, CopyAssignment) 
{
    Tensor original(shape2d, data_2x3);
    Tensor copy;
    copy = original;
    
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_NE(copy.data(), original.data()); // different memory addresses
}

TEST_F(TensorTest, SelfAssignment) 
{
    Tensor t(shape2d, data_2x3);
    float* original_ptr = t.data();
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wself-assign-overloaded"
    t = t; // self assignment
    #pragma clang diagnostic pop
    
    EXPECT_EQ(t.data(), original_ptr); // should not change
}

// testing move operations
TEST_F(TensorTest, MoveConstructor) 
{
    Tensor original(shape2d, data_2x3);
    float* original_data = original.data();
    Tensor moved(std::move(original));
    
    EXPECT_EQ(moved.shape(), shape2d);
    EXPECT_EQ(moved.size(), 6U);
    EXPECT_EQ(moved.data(), original_data); // should have the same pointer
}

TEST_F(TensorTest, MoveAssignment) 
{
    Tensor original(shape2d, data_2x3);
    float* original_data = original.data();
    Tensor moved;
    moved = std::move(original);
    
    EXPECT_EQ(moved.shape(), shape2d);
    EXPECT_EQ(moved.data(), original_data); // should have the same pointer
}

// testing element access
TEST_F(TensorTest, ElementAccess) 
{
    Tensor t(shape2d, data_2x3);
    
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST_F(TensorTest, ElementModification) 
{
    Tensor t(shape2d, data_2x3);
    t.at({0, 0}) = 42.0f;
    
    EXPECT_FLOAT_EQ(t.at({0, 0}), 42.0f);
}

TEST_F(TensorTest, ConstElementAccess) 
{
    const Tensor t(shape2d, data_2x3);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
}

TEST_F(TensorTest, OutOfBoundsAccess) 
{
    Tensor t(shape2d);
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t.at({0, 3}), std::out_of_range);
}

TEST_F(TensorTest, WrongRankAccess) 
{
    Tensor t(shape2d);
    EXPECT_THROW(t.at({0}), std::invalid_argument);      // too few indices
    EXPECT_THROW(t.at({0, 0, 0}), std::invalid_argument); // too many indices
}

// Test reshaping
TEST_F(TensorTest, ValidReshape) 
{
    Tensor t(shape2d, data_2x3);
    t.reshape({3, 2});
    
    EXPECT_EQ(t.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(t.size(), 6U); // same total size
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f); // data should be preserved
}

TEST_F(TensorTest, InvalidReshape) 
{
    Tensor t(shape2d, data_2x3);
    EXPECT_THROW(t.reshape({2, 4}), std::invalid_argument); // different total size
}

// testing arithmetic operations
TEST_F(TensorTest, Addition) 
{
    Tensor a(shape2d, data_2x3);
    Tensor b(shape2d, data_2x3);
    
    a += b;
    
    for (size_t i = 0; i < a.size(); ++i) 
    {
        EXPECT_FLOAT_EQ(a.data()[i], 2.0f * data_2x3[i]);
    }
}

TEST_F(TensorTest, AdditionOperator) 
{
    Tensor a(shape2d, data_2x3);
    Tensor b(shape2d, data_2x3);
    
    Tensor result = a + b;
    
    for (size_t i = 0; i < result.size(); ++i) 
    {
        EXPECT_FLOAT_EQ(result.data()[i], 2.0f * data_2x3[i]);
    }
}

TEST_F(TensorTest, ShapeMismatchArithmetic) 
{
    Tensor a(shape2d, data_2x3);
    Tensor b(shape3d);
    
    EXPECT_THROW(a += b, std::invalid_argument);
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST_F(TensorTest, DivisionByZero) 
{
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2, 2}, {1.0f, 0.0f, 3.0f, 4.0f}); // contains zero
    
    EXPECT_THROW(a /= b, std::invalid_argument);
}

// testing memory management (no leaks)
TEST_F(TensorTest, MultipleOperations) 
{
    // ensures no memory leaks during multiple operations
    for (int i = 0; i < 100; ++i) 
    {
        Tensor t(shape2d, data_2x3);
        Tensor copy = t;
        copy += t;
        copy.reshape({3, 2});
    }
    // if no crashes, memory management is working
    SUCCEED();
}

// perf test
TEST_F(TensorTest, LargeTensorOperations) 
{
    std::vector<size_t> large_shape = {100, 100};
    Tensor a(large_shape);
    Tensor b(large_shape);
    
    for (size_t i = 0; i < a.size(); ++i) 
    {
        a.data()[i] = static_cast<float>(i);
        b.data()[i] = static_cast<float>(i * 2);
    }
    
    Tensor result = a + b;
    
    EXPECT_FLOAT_EQ(result.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(result.data()[1], 3.0f); // 1 + 2
    EXPECT_FLOAT_EQ(result.data()[10], 30.0f); // 10 + 20
}