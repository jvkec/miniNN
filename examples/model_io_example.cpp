/* model_io_example.cpp
 * 
 * Example demonstrating how to save and load models in miniNN.
 * This example:
 * 1. Creates a simple model
 * 2. Saves it to a file
 * 3. Loads it back
 * 4. Verifies the loaded model produces identical results
 */

#include "inference_engine.h"
#include "model_loader.h"
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace mininn;

// Utility to compare tensor values
bool tensorsEqual(const Tensor& t1, const Tensor& t2, float epsilon = 1e-5f) {
    if (t1.shape() != t2.shape()) return false;
    
    for (size_t i = 0; i < t1.size(); ++i) {
        if (std::abs(t1.data()[i] - t2.data()[i]) > epsilon) return false;
    }
    return true;
}

int main() {
    try {
        std::cout << "miniNN Model I/O Example\n";
        std::cout << "=======================\n\n";

        // 1. Create a simple model
        std::cout << "Creating model...\n";
        auto model = std::make_unique<Model>();
        
        // Simple network: Linear(2->3) -> ReLU -> Linear(3->2)
        Tensor weights1({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        Tensor bias1({3}, {0.1f, 0.2f, 0.3f});
        auto linear1 = std::make_unique<LinearLayer>(weights1, bias1);
        
        auto relu = std::make_unique<ReLULayer>();
        
        Tensor weights2({3, 2}, {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f});
        Tensor bias2({2}, {0.01f, 0.02f});
        auto linear2 = std::make_unique<LinearLayer>(weights2, bias2);
        
        model->addLayer(std::move(linear1));
        model->addLayer(std::move(relu));
        model->addLayer(std::move(linear2));
        
        model->setInputShape({2});
        model->setOutputShape({2});
        
        // 2. Save the model
        const std::string model_path = "models/test_model.bin";
        std::cout << "Saving model to " << model_path << "...\n";
        ModelLoader::saveToFile(*model, model_path);
        
        // Create inference engine for the original model
        InferenceEngine original_engine(std::move(model));
        
        // 3. Load the model back
        std::cout << "Loading model from " << model_path << "...\n";
        auto loaded_model = ModelLoader::loadFromFile(model_path);
        InferenceEngine loaded_engine(std::move(loaded_model));
        
        // 4. Verify both models produce identical results
        std::cout << "Verifying model...\n";
        
        // Test with different inputs
        std::vector<std::vector<float>> test_inputs = {
            {1.0f, 2.0f},
            {-1.0f, 0.5f},
            {0.0f, 0.0f}
        };
        
        bool all_passed = true;
        for (size_t i = 0; i < test_inputs.size(); ++i) {
            const auto& input_data = test_inputs[i];
            Tensor input({2}, input_data);
            
            // Run both models
            Tensor output_original = original_engine.predict(input);
            Tensor output_loaded = loaded_engine.predict(input);
            
            // Compare results
            bool passed = tensorsEqual(output_original, output_loaded);
            all_passed &= passed;
            
            std::cout << "\nTest " << (i + 1) << ":\n";
            std::cout << "Input: [" << input_data[0] << ", " << input_data[1] << "]\n";
            std::cout << "Original output: [";
            for (size_t j = 0; j < output_original.size(); ++j) {
                std::cout << output_original.data()[j];
                if (j < output_original.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            std::cout << "Loaded output:   [";
            for (size_t j = 0; j < output_loaded.size(); ++j) {
                std::cout << output_loaded.data()[j];
                if (j < output_loaded.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            std::cout << (passed ? "✓ PASS" : "✗ FAIL") << "\n";
        }
        
        std::cout << "\nOverall: " << (all_passed ? "✓ All tests passed!" : "✗ Some tests failed!") << "\n";
        return all_passed ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
