#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>

namespace mininn 
{
    // TODO later: add more data type support
    enum class DataType 
    {
        FLOAT32,
        INT8,
        INT4
    };

    class Tensor 
    {
    public:
        // constructors
        Tensor();
        explicit Tensor(const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32);
        Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, 
               DataType dtype = DataType::FLOAT32);
        
        // deep copy constructor and assignment operator
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        
        // move constructor and assignment operator
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        
        // destructor
        ~Tensor() = default;

        // accessors
        const std::vector<size_t>& shape() const { return shape_; }
        size_t rank() const { return shape_.size(); }
        size_t size() const { return total_size_; }
        DataType dtype() const { return dtype_; }
        
        // data access
        float* data() { return data_.get(); }
        const float* data() const { return data_.get(); }
        
        // element access with bounds checking
        float& at(const std::vector<size_t>& indices);
        const float& at(const std::vector<size_t>& indices) const;
        
        void reshape(const std::vector<size_t>& new_shape);
        
        Tensor& operator+=(const Tensor& other);
        Tensor& operator-=(const Tensor& other);
        Tensor& operator*=(const Tensor& other);
        Tensor& operator/=(const Tensor& other);

    private:
        std::vector<size_t> shape_;        // shape of the tensor (e.g. [2,3,4] for 2x3x4 tensor)
        size_t total_size_;                // total number of elements
        DataType dtype_;
        std::unique_ptr<float[]> data_;    // actual data storage using smart pointer
        
        // helper methods
        void validateShape(const std::vector<size_t>& shape) const;
        size_t calculateIndex(const std::vector<size_t>& indices) const;
        size_t calculateTotalSize() const;
    };

    // binary operators
    Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    Tensor operator-(const Tensor& lhs, const Tensor& rhs);
    Tensor operator*(const Tensor& lhs, const Tensor& rhs);
    Tensor operator/(const Tensor& lhs, const Tensor& rhs);

} // namespace mininn
