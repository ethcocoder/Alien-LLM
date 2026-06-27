#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "orchestrator.hpp"

int main() {
    std::cout << "Starting Checkpoint Verification Test..." << std::endl;

    int vocab_size = 1000;
    int d_model = 256;
    std::string test_path = "test_checkpoint.bin";

    // 1. Create first instance and generate baseline
    AI2Orchestrator model1(vocab_size, d_model);
    Eigen::VectorXf task_emb = Eigen::VectorXf::Random(16);
    int test_token = 42;

    std::cout << "Generating baseline with Model 1..." << std::endl;
    Eigen::VectorXf out1 = model1.process_token(test_token, task_emb);

    // 2. Save Model 1
    std::cout << "Saving Model 1 to " << test_path << "..." << std::endl;
    model1.save_checkpoint(test_path);

    // 3. Create second instance and load checkpoint
    std::cout << "Initializing Model 2 and loading checkpoint..." << std::endl;
    AI2Orchestrator model2(vocab_size, d_model);
    model2.load_checkpoint(test_path);

    // 4. Generate output with Model 2
    std::cout << "Generating output with Model 2..." << std::endl;
    model2.reset(); // Ensure state is fresh
    model1.reset(); // Reset model 1 too just in case
    
    // Reprocess in model 2
    Eigen::VectorXf out2 = model2.process_token(test_token, task_emb);

    // 5. Compare outputs
    float diff = (out1 - out2).norm();
    std::cout << "Difference norm: " << diff << std::endl;

    if (diff < 1e-6f) {
        std::cout << "SUCCESS: Checkpoint load/save verified! Outputs match." << std::endl;
    } else {

        std::cerr << "FAILURE: Outputs do not match! Checkpoint system is still broken." << std::endl;
        return 1;
    }

    return 0;
}
