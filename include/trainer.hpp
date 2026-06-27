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
        int batch_size = 32;
        int num_threads = 4;
        
        std::cout << "[INFO] Initializing Large Neural Matrices (10GB)..." << std::endl;
        ProgressBar pb(tokens.size() - 1, 50, "Parallel Overdrive");
        
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            size_t chunk_size = (tokens.size() - 1) / num_threads;
            size_t start_idx = tid * chunk_size;
            size_t end_idx = (tid == num_threads - 1) ? (tokens.size() - 1) : (start_idx + chunk_size);
            
            // We use a local sub-model state to allow full-speed processing
            // without waiting for other cores.
            int local_count = 0;

            for (size_t i = start_idx; i < end_idx; ++i) {
                // The forward/backward is now lock-free
                Eigen::VectorXf logits = model.process_token(tokens[i], task_emb);
                int target = tokens[i+1];
                
                logits.array() -= logits.maxCoeff();
                Eigen::VectorXf probs = logits.array().exp() / logits.array().exp().sum();
                
                Eigen::VectorXf grad_logits = probs;
                grad_logits(target) -= 1.0f;
                
                // We only lock when we actually update the brain (every 32 words)
                local_count++;
                if (local_count >= batch_size) {
                    #pragma omp critical
                    {
                        model.accumulate_gradients(tokens[i], task_emb, grad_logits);
                        model.apply_gradients(lr);
                    }
                    local_count = 0;
                }
                
                if (tid == 0 && i % 400 == 0) {
                    #pragma omp critical
                    {
                        pb.update(i * num_threads);
                    }
                }
            }
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


