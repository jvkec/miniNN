/* mnist_inference_example.cpp
 * 
 * Example demonstrating MNIST digit classification using miniNN.
 * Creates and runs a simple CNN-like network structure:
 * Input (28x28) -> Linear (784->128) -> ReLU -> Linear (128->10) -> Softmax -> Output (10)
 */

#include "inference_engine.h"
#include "model_loader.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

using namespace mininn;

// Utility function to generate a sample digit (simulated MNIST data)
Tensor generateSampleDigit(int digit, std::mt19937& rng) {
    std::vector<float> pixel_data(28 * 28, 0.0f);  // 28x28 image
    std::normal_distribution<float> noise(0.0f, 0.1f);

    // Create a simple pattern for each digit
    // This is a very simplified version - real MNIST is much more complex
    switch (digit) {
        case 0: {  // Draw a circle-like pattern
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    float dx = i - 13.5f;
                    float dy = j - 13.5f;
                    float dist = std::sqrt(dx*dx + dy*dy);
                    if (dist > 8.0f && dist < 11.0f) {
                        pixel_data[i*28 + j] = 1.0f + noise(rng);
                    }
                }
            }
            break;
        }
        case 1: {  // Vertical line
            for (int i = 7; i < 21; i++) {
                for (int j = 13; j < 15; j++) {
                    pixel_data[i*28 + j] = 1.0f + noise(rng);
                }
            }
            break;
        }
        // Add more digits as needed...
        default: {  // Random pattern for other digits
            for (int i = 0; i < 784; i++) {
                pixel_data[i] = std::abs(noise(rng));
            }
        }
    }

    return Tensor({28 * 28}, pixel_data);
}



int main() {
    try {
        std::cout << "miniNN MNIST Inference Example\n";
        std::cout << "============================\n\n";

        // Create a simple model for MNIST classification
        auto model = std::make_unique<Model>();

        // Layer 1: Linear transformation (784 -> 128)
        std::vector<float> weights1(784 * 128);
        std::vector<float> bias1(128);
        
        // Initialize with Xavier/Glorot initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale1 = std::sqrt(2.0f / (784.0f + 128.0f));
        std::normal_distribution<float> dist1(0.0f, scale1);
        
        for (auto& w : weights1) w = dist1(gen);
        for (auto& b : bias1) b = 0.0f;

        Tensor weights1_tensor({784, 128}, weights1);
        Tensor bias1_tensor({128}, bias1);
        auto linear1 = std::make_unique<LinearLayer>(weights1_tensor, bias1_tensor);

        // ReLU activation
        auto relu = std::make_unique<ReLULayer>();

        // Layer 2: Linear transformation (128 -> 10)
        std::vector<float> weights2(128 * 10);
        std::vector<float> bias2(10);
        
        float scale2 = std::sqrt(2.0f / (128.0f + 10.0f));
        std::normal_distribution<float> dist2(0.0f, scale2);
        
        for (auto& w : weights2) w = dist2(gen);
        for (auto& b : bias2) b = 0.0f;

        Tensor weights2_tensor({128, 10}, weights2);
        Tensor bias2_tensor({10}, bias2);
        auto linear2 = std::make_unique<LinearLayer>(weights2_tensor, bias2_tensor);

        // Softmax for output probabilities
        auto softmax = std::make_unique<SoftmaxLayer>();

        // Add layers to model
        model->addLayer(std::move(linear1));
        model->addLayer(std::move(relu));
        model->addLayer(std::move(linear2));
        model->addLayer(std::move(softmax));

        // Set model input/output shapes
        model->setInputShape({784});  // 28x28 = 784
        model->setOutputShape({10});  // 10 digits (0-9)

        // Save the model (this would typically be done after training)
        const std::string model_path = "models/mnist_model.bin";
        std::cout << "\nSaving model to " << model_path << "...\n";
        ModelLoader::saveToFile(*model, model_path);
        
        // In a real application, you would load a pre-trained model:
        // auto trained_model = ModelLoader::loadFromFile("path/to/pretrained/model.bin");
        // But for this example, we'll use our randomly initialized model
        
        // Create inference engine
        InferenceEngine engine(std::move(model));
        engine.enableProfiling(true);

        std::cout << "Model architecture:\n";
        std::cout << "- Input: 784 neurons (28x28 image)\n";
        std::cout << "- Hidden layer: 128 neurons with ReLU\n";
        std::cout << "- Output: 10 neurons with Softmax\n\n";

        // Generate and test some sample digits
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        
        std::cout << "Running inference on sample digits:\n";
        std::cout << std::fixed << std::setprecision(4);

        // Test digits 0 and 1 (could extend to more)
        for (int digit : {0, 1}) {
            std::cout << "\nTesting digit " << digit << ":\n";
            
            // Generate sample digit
            Tensor input = generateSampleDigit(digit, rng);
            
            // Normalize input (assuming MNIST-like normalization)
            input = InferenceUtils::normalizeInput(input, 0.5f, 0.5f);
            

            
            // Run inference
            Tensor output = engine.predict(input);
            
            // Print probabilities
            std::cout << "\nPredicted probabilities:\n";
            for (size_t i = 0; i < output.size(); ++i) {
                std::cout << "  Digit " << i << ": " << output.data()[i] << "\n";
            }
            
            // Get top prediction
            size_t predicted = InferenceUtils::getArgMax(output);
            std::cout << "\nPredicted digit: " << predicted << "\n";
            
            // Get top-3 predictions
            auto top3 = InferenceUtils::getTopK(output, 3);
            std::cout << "Top 3 predictions:\n";
            for (const auto& [idx, prob] : top3) {
                std::cout << "  Digit " << idx << ": " << prob << "\n";
            }
        }

        // Print profiling information
        const auto& stats = engine.getLastInferenceStats();
        std::cout << "\nProfiling Information:\n";
        std::cout << "  Total inference time: " << stats.total_time.count() << " ms\n";
        std::cout << "  Layer timings:\n";
        for (size_t i = 0; i < stats.layer_times.size(); ++i) {
            std::cout << "    Layer " << i << ": " << stats.layer_times[i].count() << " ms\n";
        }

        std::cout << "\n✓ MNIST example completed successfully!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
