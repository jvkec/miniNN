/* simple_inference_example.cpp
 * 
 * A simple example demonstrating how to create a model programmatically
 * and run inference using the miniNN inference engine.
 * 
 * This example creates a simple 2-layer neural network:
 * Input (2) -> Linear (2->3) -> ReLU -> Output (3)
 */

#include "inference_engine.h"
#include "model_loader.h"
#include <iostream>
#include <iomanip>

using namespace mininn;

int main() 
{
    try 
    {
        std::cout << "miniNN Inference Engine Example\n";
        std::cout << "================================\n\n";

        // create a simple model programmatically
        auto model = std::make_unique<Model>();
        
        // layer 1: Linear transformation (2 inputs -> 3 outputs)
        // weights: [1, 2, 3]  (first row: weights from input[0] to outputs)
        //          [4, 5, 6]  (second row: weights from input[1] to outputs)
        Tensor weights({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        Tensor bias({3}, {0.1f, 0.2f, 0.3f});
        
        auto linear_layer = std::make_unique<LinearLayer>(weights, bias);
        
        // layer 2: ReLU activation
        auto relu_layer = std::make_unique<ReLULayer>();
        
        // add layers to model
        model->addLayer(std::move(linear_layer));
        model->addLayer(std::move(relu_layer));
        
        // set model input/output shapes
        model->setInputShape({2});
        model->setOutputShape({3});
        
        std::cout << "Created model with:\n";
        std::cout << "- Input shape: [2]\n";
        std::cout << "- Layer 1: Linear (2 -> 3)\n";
        std::cout << "- Layer 2: ReLU activation\n";
        std::cout << "- Output shape: [3]\n\n";
        
        // create inference engine
        InferenceEngine engine(std::move(model));
        engine.enableProfiling(true);
        
        std::cout << "Model loaded successfully!\n";
        std::cout << "Number of layers: " << engine.getNumLayers() << "\n\n";
        
        // test different inputs
        std::vector<std::vector<float>> test_inputs = {
            {1.0f, 2.0f},
            {0.5f, 1.5f},
            {-1.0f, 3.0f},
            {0.0f, 0.0f}
        };
        
        std::cout << "Running inference on test inputs:\n";
        std::cout << std::fixed << std::setprecision(3);
        
        for (size_t i = 0; i < test_inputs.size(); ++i) 
        {
            const auto& input_data = test_inputs[i];
            Tensor input({2}, input_data);
            
            std::cout << "\nTest " << (i + 1) << ":\n";
            std::cout << "  Input:  [" << input_data[0] << ", " << input_data[1] << "]\n";
            
            // perform inference
            Tensor output = engine.predict(input);
            
            std::cout << "  Output: [";
            for (size_t j = 0; j < output.size(); ++j) 
            {
                std::cout << output.data()[j];
                if (j < output.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // manual calculation for verification:
            // Linear: out = input * weights + bias
            // out[0] = input[0]*1 + input[1]*4 + 0.1
            // out[1] = input[0]*2 + input[1]*5 + 0.2  
            // out[2] = input[0]*3 + input[1]*6 + 0.3
            // then ReLU: max(0, out)
            
            float expected[3];
            expected[0] = std::max(0.0f, input_data[0]*1.0f + input_data[1]*4.0f + 0.1f);
            expected[1] = std::max(0.0f, input_data[0]*2.0f + input_data[1]*5.0f + 0.2f);
            expected[2] = std::max(0.0f, input_data[0]*3.0f + input_data[1]*6.0f + 0.3f);
            
            std::cout << "  Expected: [" << expected[0] << ", " << expected[1] << ", " << expected[2] << "]\n";
            
            // verify accuracy
            bool accurate = true;
            for (size_t j = 0; j < 3; ++j) 
            {
                if (std::abs(output.data()[j] - expected[j]) > 1e-5) 
                {
                    accurate = false;
                    break;
                }
            }
            std::cout << "  ✓ " << (accurate ? "PASS" : "FAIL") << "\n";
        }
        
        // profiling information
        const auto& stats = engine.getLastInferenceStats();
        std::cout << "\nProfiling Information:\n";
        std::cout << "  Total inference time: " << stats.total_time.count() << " ms\n";
        std::cout << "  Layer timings:\n";
        for (size_t i = 0; i < stats.layer_times.size(); ++i) 
        {
            std::cout << "    Layer " << i << ": " << stats.layer_times[i].count() << " ms\n";
        }
        
        // demo utility functions
        std::cout << "\nUtility Functions Demo:\n";
        Tensor scores({4}, {0.1f, 0.8f, 0.3f, 0.2f});
        
        // find argmax
        size_t max_idx = InferenceUtils::getArgMax(scores);
        std::cout << "  Argmax of [0.1, 0.8, 0.3, 0.2]: " << max_idx << "\n";
        
        // get top-2
        auto top_2 = InferenceUtils::getTopK(scores, 2);
        std::cout << "  Top-2: ";
        for (const auto& pair : top_2) 
        {
            std::cout << "(" << pair.first << ":" << pair.second << ") ";
        }
        std::cout << "\n";
        
        // normalization
        Tensor data({3}, {2.0f, 4.0f, 6.0f});
        Tensor normalized = InferenceUtils::normalizeInput(data, 4.0f, 2.0f);
        std::cout << "  Normalized [2,4,6] with mean=4, std=2: [";
        for (size_t i = 0; i < normalized.size(); ++i) 
        {
            std::cout << normalized.data()[i];
            if (i < normalized.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        std::cout << "\n✓ Example completed successfully!\n";
        return 0;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
