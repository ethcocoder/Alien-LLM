#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "tokenizer.hpp"
#include "orchestrator.hpp"
#include "dataloader.hpp"
#include "utils.hpp"

extern "C" {
    typedef struct {
        Tokenizer* tokenizer;
        AI2Orchestrator* model;
        Eigen::VectorXf* task_emb;
    } ModelInstance;

    ModelInstance* create_model(const char* checkpoint_path, int d_model_ignored) {
        int d_model = 256; 
        std::cout << "[BACKEND] Initializing Alien-LLM Model..." << std::endl;
        
        ModelInstance* instance = new ModelInstance();
        instance->tokenizer = new Tokenizer();
        
        // Train tokenizer to ensure vocab is ready
        std::string dataset_path = "datasets/TinyStories-valid.txt";
        std::cout << "[BACKEND] Loading tokenizer dataset from: " << dataset_path << std::endl;
        
        std::string tinystories_text = DataLoader::load_text(dataset_path, 10000);
        if (tinystories_text.empty()) {
            std::cerr << "[ERROR] Dataset not found or empty! Tokenizer training may fail." << std::endl;
            // Fallback to minimal text to prevent crash if possible
            tinystories_text = "The quick brown fox jumps over the lazy dog.";
        }
        
        std::cout << "[BACKEND] Training tokenizer..." << std::endl;
        instance->tokenizer->train(tinystories_text);
        std::cout << "[BACKEND] Tokenizer ready. Vocab size: " << instance->tokenizer->vocab_size() << std::endl;

        instance->model = new AI2Orchestrator(instance->tokenizer->vocab_size(), d_model);
        
        // Enable checkpoint loading
        std::cout << "[BACKEND] Checking checkpoint at: " << checkpoint_path << std::endl;
        std::ifstream f(checkpoint_path, std::ios::binary);
        if (f.good()) {
            std::cout << "[BACKEND] Loading checkpoint..." << std::endl;
            instance->model->load_checkpoint(checkpoint_path);
            std::cout << "[BACKEND] Checkpoint loaded successfully." << std::endl;
        } else {
            std::cerr << "[WARNING] Checkpoint not found at " << checkpoint_path << ". Starting with random weights." << std::endl;
        }
        
        instance->task_emb = new Eigen::VectorXf(16);
        *instance->task_emb = Eigen::VectorXf::Random(16);
        
        std::cout << "[BACKEND] Model initialization complete." << std::endl;
        return instance;
    }

    void destroy_model(ModelInstance* instance) {
        delete instance->tokenizer;
        delete instance->model;
        delete instance->task_emb;
        delete instance;
    }

    void reset_model(ModelInstance* instance) {
        instance->model->reset();
    }

    const char* generate_token(ModelInstance* instance, int last_token_id) {
        Eigen::VectorXf out = instance->model->process_token(last_token_id, *instance->task_emb);
        
        // Simplified next token selection logic from main.cpp
        int next_id = (int)(std::abs(out.sum())) % instance->tokenizer->vocab_size();
        if (next_id < 4) next_id = 4 + (rand() % (instance->tokenizer->vocab_size() - 4));
        
        static std::string last_decoded_token;
        last_decoded_token = instance->tokenizer->decode({next_id});
        
        return last_decoded_token.c_str();
    }

    int encode_text(ModelInstance* instance, const char* text, int* tokens) {
        std::vector<int> encoded = instance->tokenizer->encode(text);
        for (size_t i = 0; i < encoded.size(); ++i) {
            tokens[i] = encoded[i];
        }
        return (int)encoded.size();
    }

    void process_tokens(ModelInstance* instance, int* tokens, int count) {
        ProgressBar pb(count, 30, "Pre-processing");
        for (int i = 0; i < count; ++i) {
            instance->model->process_token(tokens[i], *instance->task_emb);
            if (i % 10 == 0) pb.update(i);
        }
        pb.update(count);
    }

    int get_last_token_id(ModelInstance* instance, const char* text) {
        std::vector<int> encoded = instance->tokenizer->encode(text);
        if (encoded.empty()) return 2; // BOS
        return encoded.back();
    }
}


