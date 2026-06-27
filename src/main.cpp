#include <iostream>
#include <vector>
#include <string>
#include "tokenizer.hpp"
#include "orchestrator.hpp"
#include "trainer.hpp"
#include "dataloader.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {

    std::cout << "--- Alien Intelligence (AI2) Training Pipeline ---" << std::endl;

    bool resume = (argc > 1 && std::string(argv[1]) == "--resume");
    bool quick_mode = (argc > 1 && std::string(argv[1]) == "--quick") || (argc > 2 && std::string(argv[2]) == "--quick");
    
    std::string checkpoint_path = "model_checkpoint.bin";

    // 1. Data Loading
    int data_limit = quick_mode ? 5000 : 500000;
    std::cout << "[INFO] Loading dataset with limit: " << data_limit << " characters." << std::endl;
    
    std::string tinystories_text = DataLoader::load_text("datasets/TinyStories-valid.txt", data_limit);
    auto alpaca_raw = DataLoader::load_jsonl("datasets/alpaca_data.jsonl", quick_mode ? 50 : 500);


    // 2. Tokenizer
    if (tinystories_text.empty()) {
        std::cerr << "[FATAL] Dataset is empty! Please check if 'datasets/TinyStories-valid.txt' exists." << std::endl;
        return 1;
    }

    Tokenizer tokenizer;
    tokenizer.train(tinystories_text);
    for (const auto& pair : alpaca_raw) {
        tokenizer.train(pair.first);
        tokenizer.train(pair.second);
    }

    // 3. Model
    if (tokenizer.vocab_size() == 0) {
        std::cerr << "[FATAL] Tokenizer training failed (vocab size 0)." << std::endl;
        return 1;
    }

    int d_model = 256;
    AI2Orchestrator model(tokenizer.vocab_size(), d_model);
    if (resume) {
        std::cout << "Resuming from checkpoint: " << checkpoint_path << std::endl;
        model.load_checkpoint(checkpoint_path);
    }

    AI2Trainer trainer(model, 0.002f);

    Eigen::VectorXf task_emb = Eigen::VectorXf::Random(16);

    // 4. Training Loop (Epochs)
    int epochs = 5;
    std::cout << "Starting training for " << epochs << " epochs..." << std::endl;
    
    std::vector<int> train_tokens = tokenizer.encode(tinystories_text);
    if (train_tokens.empty()) {
        std::cerr << "[FATAL] Tokenization failed." << std::endl;
        return 1;
    }

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::cout << "\n--- Epoch " << epoch << "/" << epochs << " ---" << std::endl;
        trainer.train_step(train_tokens, task_emb);
        
        // Save checkpoint after every epoch
        model.save_checkpoint(checkpoint_path);
        std::cout << "Epoch " << epoch << " complete. Checkpoint saved." << std::endl;
    }



    // 6. Validation & Logging
    std::cout << "\n--- Communication Validation ---" << std::endl;
    std::vector<std::string> prompts = {
        "Once upon a time, there was a little bird who",
        "Instruction: Explain the concept of gravity.\nInput: \nResponse: ",
        "Instruction: Write a short poem about the moon.\nInput: \nResponse: "
    };

    std::ofstream log_file("communication_log.txt");
    for (const auto& prompt : prompts) {
        std::cout << "User: " << prompt << std::endl;
        log_file << "User: " << prompt << "\n";
        
        std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        model.reset();
        for (int id : prompt_tokens) model.process_token(id, task_emb);

        std::cout << "AI2: ";
        log_file << "AI2: ";
        int current_id = prompt_tokens.back();
        ProgressBar pg(20, 30, "Generating");
        for (int i = 0; i < 20; ++i) {
            Eigen::VectorXf logits = model.process_token(current_id, task_emb);
            
            // Softmax with Temperature
            float temp = 0.8f;
            logits.array() -= logits.maxCoeff();
            Eigen::VectorXf probs = (logits.array() / temp).exp();
            probs /= probs.sum();
            
            // Random weighted selection
            float r = (float)rand() / (float)RAND_MAX;
            int next_id = 0;
            float cumsum = 0;
            for (int j = 0; j < probs.size(); ++j) {
                cumsum += probs(j);
                if (r <= cumsum) {
                    next_id = j;
                    break;
                }
            }
            
            if (next_id < 4) next_id = 4 + (rand() % (tokenizer.vocab_size() - 4));
            
            std::string word = tokenizer.decode({next_id});
            log_file << word << " ";
            current_id = next_id;
            pg.update(i + 1);
        }

        std::cout << "\n" << std::endl;

        log_file << "\n\n";
    }
    log_file.close();

    std::cout << "Pipeline complete. Logs saved to communication_log.txt" << std::endl;
    return 0;
}

