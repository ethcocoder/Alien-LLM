#ifndef ENSEMBLE_HPP
#define ENSEMBLE_HPP

#include <vector>
#include <memory>
#include <future>
#include <Eigen/Dense>
#include "orchestrator.hpp"

class MixtureOfExperts {
public:
    MixtureOfExperts(int n_experts, int vocab_size, int d_model) 
        : n_experts(n_experts) {
        for (int i = 0; i < n_experts; ++i) {
            experts.push_back(std::make_unique<AI2Orchestrator>(vocab_size, d_model));
        }
        // Gating network: simple linear layer to weight experts
        W_gate = Eigen::MatrixXf::Random(n_experts, d_model);
    }

    Eigen::VectorXf forward(int token_id, const Eigen::VectorXf& task_emb, const Eigen::VectorXf& current_hidden) {
        // 1. Compute expert weights
        Eigen::VectorXf gate_logits = W_gate * current_hidden;
        Eigen::VectorXf weights = gate_logits.array().exp() / gate_logits.array().exp().sum();

        // 2. Run experts in parallel
        std::vector<std::future<Eigen::VectorXf>> futures;
        for (int i = 0; i < n_experts; ++i) {
            futures.push_back(std::async(std::launch::async, [&, i]() {
                return experts[i]->process_token(token_id, task_emb);
            }));
        }

        // 3. Aggregate results
        Eigen::VectorXf output = Eigen::VectorXf::Zero(experts[0]->get_d_model());
        for (int i = 0; i < n_experts; ++i) {
            output += weights(i) * futures[i].get();
        }

        return output;
    }

    void load_experts(const std::vector<std::string>& paths) {
        for (size_t i = 0; i < std::min(paths.size(), experts.size()); ++i) {
            experts[i]->load_checkpoint(paths[i]);
        }
    }

    void reset() {
        for (auto& expert : experts) {
            expert->reset();
        }
    }

private:
    int n_experts;
    std::vector<std::unique_ptr<AI2Orchestrator>> experts;
    Eigen::MatrixXf W_gate;
};

#endif // ENSEMBLE_HPP
