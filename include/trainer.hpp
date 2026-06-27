#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <vector>
#include <iostream>
#include "orchestrator.hpp"
#include "utils.hpp"


class AI2Trainer {
public:
    AI2Trainer(AI2Orchestrator& model, float lr = 0.01f) : model(model), lr(lr) {}

    void train_step(const std::vector<int>& tokens, const Eigen::VectorXf& task_emb) {
        model.reset();
        ProgressBar pb(tokens.size() - 1, 50, "Training");
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            Eigen::VectorXf logits = model.process_token(tokens[i], task_emb);
            int target = tokens[i+1];
            
            // 1. Stable Softmax
            logits.array() -= logits.maxCoeff();
            Eigen::VectorXf probs = logits.array().exp() / logits.array().exp().sum();

            
            // 2. Cross-Entropy Loss (NLL)
            float loss = -std::log(std::max(probs(target), 1e-9f));
            
            // 3. Backward (Gradient of Cross-Entropy w.r.t Logits)
            Eigen::VectorXf grad_logits = probs;
            grad_logits(target) -= 1.0f;
            
            // 4. Update Weights
            model.update(tokens[i], task_emb, grad_logits, lr);
            
            if (i % 100 == 0) pb.update(i);
        }
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
                
                model.update(input_token, task_emb, grad_logits, lr);
            }

        }
        std::cout << "Fine-tuning complete." << std::endl;
    }

private:
    AI2Orchestrator& model;
    float lr;
};

#endif // TRAINER_HPP


