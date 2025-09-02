#pragma once
#include "tensor.h"

namespace mininn
{
    class TensorOps
    {
    public:

        // static methods since no class instance is required -> idiomatic in c++
        static void matmul(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);

        // cache friendly version
        // static void matmul_optimized(const Tensor& tensor1, const Tensor& tensor2, Tensor& result);

        static void relu(Tensor& tensor);
        static void sigmoid(Tensor& tensor);
        static void softmax(Tensor& tensor);
    };
}; // namespace mininn