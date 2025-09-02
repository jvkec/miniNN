# miniNN
Simplified neural network inference engine inspired by llama.cpp. Learning project for me!

## Included Stuff

### Core Features
- **Tensor operations**: Matrix multiplication, element-wise operations
- **Activation functions**: ReLU, Sigmoid, Softmax
- **Layer types**: Linear (fully connected), activation layers
- **Model loading**: Custom binary `.minn` format with validation
- **Inference engine**: Forward pass execution with profiling
- **Error handling**: Comprehensive validation and clear error messages

### Implementation Details
- **Memory management**: RAII with smart pointers, no memory leaks
- **Performance**: Pre-allocated buffers, optimized matrix operations
- **Testing**: 87 unit and integration tests
- **Examples**: Simple inference, model I/O, MNIST inference demos
- **Build system**: Modern C++17, multiple build configurations

### Supported Operations
```cpp
// Tensor operations
Tensor result = tensor1 + tensor2;  // Element-wise operations
TensorOps::matmul(a, b, result);    // Matrix multiplication

// Neural network layers
LinearLayer(weights, bias);         // Fully connected
ReLULayer(), SigmoidLayer();        // Activations

// Inference
InferenceEngine engine(model);
Tensor output = engine.predict(input);
```

## Everything it doesn't include (it's a lot...):

### Training & Optimization
- **No training (its for inference)**: Forward pass only, no backpropagation
- **No optimizers**: No SGD, Adam, etc.
- **No loss functions**: Inference-only system
- **No gradient computation**: Not needed for inference

### Advanced Features  
- **No GPU support**: CPU-only implementation
- **No quantization**: Only FLOAT32 (INT8/INT4 planned)
- **No SIMD optimizations**: Basic matrix operations
- **No threading**: Single-threaded execution
- **No dynamic graphs**: Static model structure only

### Model Formats
- **No ONNX support**: Custom `.minn` format only
- **No PyTorch/TensorFlow import**: Would need converters
- **No pre-trained models**: Create your own or convert

### Layer Types
- **No convolutions**: Only fully connected layers
- **No normalization**: No BatchNorm, LayerNorm
- **No recurrent layers**: No LSTM, GRU
- **No attention**: No transformers, self-attention

**Future Development**: I do plan to further develop on this to support smaller SOTA open source models, 
adding quantization, more layer types, and model format converters :D

## Quick Start

```bash
# Build and run tests
make debug
./build/all_tests

# Run examples
make simple     # Simple inference demo  
make mnist      # MNIST inference example
make model-io   # Model save/load demo

# Build help
make help
```

## File Structure

```
miniNN/
├── src/           # Core implementation
├── include/       # Header files  
├── tests/         # Unit and integration tests
├── examples/      # Demo applications
├── models/        # Model files (.minn format)
└── build/         # Compiled binaries
```

## Requirements

- C++17 compiler (GCC/Clang)
- Google Test (for running tests)

---

*miniNN is designed for education and simple inference tasks. Consider TensorFlow Lite, ONNX Runtime, or PyTorch Mobile for the good stuff.*