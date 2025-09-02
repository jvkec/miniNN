/* tensor_ops.cpp
 * 
 * Implementation of the TensorOps class, providing core tensor operations.
 */

#include "tensor_ops.h"
#include <stdexcept>
#include <string>
#include <cmath>

namespace mininn
{
    void TensorOps::matmul(const Tensor& tensor1, const Tensor& tensor2, Tensor& result)
    {
        // tensor1 dimensions are m x n and tensor2 dimensions are n x p -> result dimensions are m x p
        if (tensor1.rank() != 2 || tensor2.rank() != 2)
        {
            throw std::invalid_argument("Matrix multiplication requires 2D tensors");
        }

        const std::vector<size_t> shape1 = tensor1.shape();  // m x n
        const std::vector<size_t> shape2 = tensor2.shape();  // n x p

        if (shape1[1] != shape2[0])
        {
            throw std::invalid_argument(
                "Inner dimensions must match for matrix multiplication: " +
                std::to_string(shape1[1]) + " != " + std::to_string(shape2[0])
            );
        }

        const size_t m = shape1[0];
        const size_t n = shape1[1];
        const size_t p = shape2[1];

        // create result tensor with correct dimensions
        result = Tensor({m, p});

        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < p; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < n; ++k)
                {
                    sum += tensor1.at({i, k}) * tensor2.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }
    }

    // void TensorOps::matmul_optimized(const Tensor& tensor1, const Tensor& tensor2, Tensor& result)
    // {
    //     // TODO: implement optimized matrix multiplication (cache friendly version)
    // }

    void TensorOps::relu(Tensor& tensor)
    {
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            if (tensor.data()[i] < 0.0f)
            {
                tensor.data()[i] = 0.0f;
            }
        }
    }

    void TensorOps::sigmoid(Tensor& tensor)
    {
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            tensor.data()[i] = 1.0f / (1.0f + std::exp(-tensor.data()[i]));
        }
    }

    void TensorOps::softmax(Tensor& tensor)
    {
        if (tensor.size() == 0)
        {
            throw std::invalid_argument("Cannot compute Softmax on empty tensor");
        }

        // find max val to prevent overflow
        float max_val = tensor.data()[0];
        for (size_t i = 1; i < tensor.size(); ++i)
        {
            max_val = std::max(max_val, tensor.data()[i]);
        }

        // exp(x - max_val) for each element then sum
        float sum = 0.0f;
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            tensor.data()[i] = std::exp(tensor.data()[i] - max_val);
            sum += tensor.data()[i];
        }

        const float inverse_sum = 1.0f / sum;
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            tensor.data()[i] *= inverse_sum;
        }
    }
} // namespace mininn