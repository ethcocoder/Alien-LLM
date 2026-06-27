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
            Eigen::VectorXf out = model.process_token(tokens[i], task_emb);
            float loss = compute_loss(out, tokens[i+1]);
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
                int target = pair.second[i];
                Eigen::VectorXf out = model.process_token(target, task_emb);
                compute_loss(out, target); // Placeholder for weight update
            }
        }
        std::cout << "Fine-tuning complete." << std::endl;
    }

private:
    AI2Orchestrator& model;
    float lr;

    float compute_loss(const Eigen::VectorXf& out, int target_token_id) {
        // Cross-entropy loss placeholder
        return 1.0f / (out.norm() + 1e-6f); // Dummy loss
    }
};

#endif // TRAINER_HPP

