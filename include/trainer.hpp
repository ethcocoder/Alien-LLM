#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <vector>
#include <iostream>
#include "orchestrator.hpp"
#include "utils.hpp"


#include <omp.h>

class AI2Trainer {
public:
    AI2Trainer(AI2Orchestrator& model, float lr = 0.01f) : model(model), lr(lr) {}

    void train_step(const std::vector<int>& tokens, const Eigen::VectorXf& task_emb) {
        model.reset();
        int batch_size = 32;
        int accumulation_count = 0;
        
        // Use all available cores for Eigen matrix math
        Eigen::setNbThreads(omp_get_max_threads());
        
        std::cout << "[INFO] Initializing Large Neural Matrices (10GB)..." << std::endl;
        ProgressBar pb(tokens.size() - 1, 50, "Neural Overdrive");

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            Eigen::VectorXf logits = model.process_token(tokens[i], task_emb);
            int target = tokens[i+1];
            
            logits.array() -= logits.maxCoeff();
            Eigen::VectorXf probs = logits.array().exp() / logits.array().exp().sum();
            
            Eigen::VectorXf grad_logits = probs;
            grad_logits(target) -= 1.0f;
            
            model.accumulate_gradients(tokens[i], task_emb, grad_logits);
            accumulation_count++;

            if (accumulation_count >= batch_size) {
                model.apply_gradients(lr);
                accumulation_count = 0;
            }
            
            if (i % 200 == 0) pb.update(i);
        }
        model.apply_gradients(lr);
        pb.update(tokens.size() - 1);
    }








    void finetune_step(const std::vector<std::pair<std::vector<int>, std::vector<int>>>& instruction_pairs, const Eigen::VectorXf& task_emb) {
        for (const auto& pair : instruction_pairs) {
            model.reset();
            // Process prompt
            for (int id : pair.first) {
                model.process_token(id, task_emb);
            }
            // Train on completion
            for (size_t i = 0; i < pair.second.size(); ++i) {
                int input_token = (i == 0) ? pair.first.back() : pair.second[i-1];
                int target = pair.second[i];
                
                Eigen::VectorXf logits = model.process_token(input_token, task_emb);
                Eigen::VectorXf probs = logits.array().exp() / logits.array().exp().sum();
                
                Eigen::VectorXf grad_logits = probs;
                grad_logits(target) -= 1.0f;
                
                model.accumulate_gradients(input_token, task_emb, grad_logits);
            }
            model.apply_gradients(lr);
        }
        std::cout << "Fine-tuning complete." << std::endl;
    }

private:
    AI2Orchestrator& model;
    float lr;
};

#endif // TRAINER_HPP


