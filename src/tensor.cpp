/* tensor.cpp
 * 
 * Implementation of the Tensor class, providing core tensor operations and memory management.
 * This implementation focuses on efficiency while maintaining safety through bounds checking
 * and proper memory management.
 */

#include "tensor.h"
#include <numeric>
#include <algorithm>
#include <sstream>

namespace mininn 
{
    Tensor::Tensor()
        : shape_{}
        , total_size_(0)
        , dtype_(DataType::FLOAT32)
        , data_(nullptr)
    {
    }

    Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype)
        : shape_(shape)
        , dtype_(dtype)
    {
        validateShape(shape);
        total_size_ = calculateTotalSize();
        data_ = std::make_unique<float[]>(total_size_);
    }

    Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, DataType dtype)
        : shape_(shape)
        , dtype_(dtype)
    {
        validateShape(shape);
        total_size_ = calculateTotalSize();
        
        if (data.size() != total_size_) 
        {
            throw std::invalid_argument("Data size does not match tensor shape");
        }
        
        data_ = std::make_unique<float[]>(total_size_);
        std::copy(data.begin(), data.end(), data_.get());
    }

    Tensor::Tensor(const Tensor& other)
        : shape_(other.shape_)
        , total_size_(other.total_size_)
        , dtype_(other.dtype_)
    {
        data_ = std::make_unique<float[]>(total_size_);
        std::copy(other.data_.get(), other.data_.get() + total_size_, data_.get());
    }

    Tensor& Tensor::operator=(const Tensor& other)
    {
        if (this != &other) 
        {
            shape_ = other.shape_;
            total_size_ = other.total_size_;
            dtype_ = other.dtype_;
            data_ = std::make_unique<float[]>(total_size_);
            // starts copying contents from other's data addresses in memory to this tensor's data addresses
            std::copy(other.data_.get(), other.data_.get() + total_size_, data_.get());
        }
        return *this;
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , total_size_(other.total_size_)
        , dtype_(other.dtype_)
        , data_(std::move(other.data_))
    {
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept
    {
        if (this != &other) 
        {
            shape_ = std::move(other.shape_);
            total_size_ = other.total_size_;
            dtype_ = other.dtype_;
            data_ = std::move(other.data_);
        }
        return *this;
    }

    // can modify returned value
    float& Tensor::at(const std::vector<size_t>& indices)
    {
        size_t idx = calculateIndex(indices);
        return data_[idx];
    }

    // cannot modify returned value
    const float& Tensor::at(const std::vector<size_t>& indices) const
    {
        size_t idx = calculateIndex(indices);
        return data_[idx];
    }

    void Tensor::reshape(const std::vector<size_t>& new_shape)
    {
        size_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), 
                                              1ULL, std::multiplies<size_t>());
        
        if (new_total_size != total_size_)
        {
            throw std::invalid_argument("New shape must preserve total number of elements");
        }
        
        shape_ = new_shape;
    }

    // validate every dimension is not zero and shape is not empty
    void Tensor::validateShape(const std::vector<size_t>& shape) const
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty");
        }
        // check if any element is zero w/ lambda function
        if (std::any_of(shape.begin(), shape.end(), 
                        [](size_t dim) { return dim == 0; }))
        {
            throw std::invalid_argument("Shape dimensions cannot be zero");
        }
    }

    // calculating index to access data (which is a flat array)
    size_t Tensor::calculateIndex(const std::vector<size_t>& indices) const
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("Number of indices must match tensor rank");
        }
        
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= shape_[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
        }
        
        size_t index = 0;
        size_t multiplier = 1;
        // cast to int to exit loop
        for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i)
        {
            index += indices[i] * multiplier;
            multiplier *= shape_[i];
        }
        
        return index;
    }

    // calculate total number of elements in tensor
    size_t Tensor::calculateTotalSize() const
    {
        return std::accumulate(shape_.begin(), shape_.end(), 
                              1ULL, std::multiplies<size_t>());
    }

    Tensor& Tensor::operator+=(const Tensor& other)
    {
        if (shape_ != other.shape_)
        {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
        
        for (size_t i = 0; i < total_size_; ++i)
        {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Tensor& Tensor::operator-=(const Tensor& other)
    {
        if (shape_ != other.shape_)
        {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }
        
        for (size_t i = 0; i < total_size_; ++i)
        {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Tensor& Tensor::operator*=(const Tensor& other)
    {
        if (shape_ != other.shape_)
        {
            throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
        }
        
        for (size_t i = 0; i < total_size_; ++i)
        {
            data_[i] *= other.data_[i];
        }
        return *this;
    }

    Tensor& Tensor::operator/=(const Tensor& other)
    {
        if (shape_ != other.shape_)
        {
            throw std::invalid_argument("Tensor shapes must match for element-wise division");
        }
        
        for (size_t i = 0; i < total_size_; ++i)
        {
            if (other.data_[i] == 0.0f)
            {
                throw std::invalid_argument("Division by zero");
            }
            data_[i] /= other.data_[i];
        }
        return *this;
    }

    Tensor operator+(const Tensor& lhs, const Tensor& rhs)
    {
        Tensor result(lhs);
        result += rhs;
        return result;
    }

    Tensor operator-(const Tensor& lhs, const Tensor& rhs)
    {
        Tensor result(lhs);
        result -= rhs;
        return result;
    }

    Tensor operator*(const Tensor& lhs, const Tensor& rhs)
    {
        Tensor result(lhs);
        result *= rhs;
        return result;
    }

    Tensor operator/(const Tensor& lhs, const Tensor& rhs)
    {
        Tensor result(lhs);
        result /= rhs;
        return result;
    }

} // namespace mininn