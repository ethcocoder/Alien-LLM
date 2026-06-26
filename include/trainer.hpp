#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <vector>
#include <iostream>
#include "orchestrator.hpp"

class AI2Trainer {
public:
    AI2Trainer(AI2Orchestrator& model, double lr = 0.01) : model(model), lr(lr) {}

    void train_step(const std::vector<int>& tokens, const Eigen::VectorXd& task_emb) {
        model.reset();
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            Eigen::VectorXd out = model.process_token(tokens[i], task_emb);
            double loss = compute_loss(out, tokens[i+1]);
            if (i % 1000 == 0) std::cout << "Step " << i << " Loss: " << loss << std::endl;
        }
    }

    void finetune_step(const std::vector<std::pair<std::vector<int>, std::vector<int>>>& instruction_pairs, const Eigen::VectorXd& task_emb) {
        for (const auto& pair : instruction_pairs) {
            model.reset();
            // Process prompt
            for (int id : pair.first) {
                model.process_token(id, task_emb);
            }
            // Train on completion
            for (size_t i = 0; i < pair.second.size(); ++i) {
                int target = pair.second[i];
                Eigen::VectorXd out = model.process_token(target, task_emb);
                compute_loss(out, target); // Placeholder for weight update
            }
        }
        std::cout << "Fine-tuning complete." << std::endl;
    }

private:
    AI2Orchestrator& model;
    double lr;

    double compute_loss(const Eigen::VectorXd& out, int target_token_id) {
        // Cross-entropy loss placeholder
        // Normally: -log(softmax(out)[target_token_id])
        return 1.0 / (out.norm() + 1e-6); // Dummy loss
    }
};

#endif // TRAINER_HPP
