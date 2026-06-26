#ifndef SSOG_HPP
#define SSOG_HPP

#include <vector>
#include <Eigen/Dense>

class SparseSynthesizer {
public:
    SparseSynthesizer(int d_in, int d_out, int n_experts, int k_active) 
        : d_in(d_in), d_out(d_out), n_experts(n_experts), k_active(k_active) {
        for (int i = 0; i < n_experts; ++i) {
            experts.push_back(Eigen::MatrixXd::Random(d_out, d_in));
        }
        W_gate = Eigen::MatrixXd::Random(n_experts, d_in);
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& h) {
        Eigen::VectorXd gate_logits = W_gate * h;
        
        // Simple top-k routing
        std::vector<int> indices(n_experts);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k_active, indices.end(),
                          [&](int i, int j) { return gate_logits(i) > gate_logits(j); });

        Eigen::VectorXd output = Eigen::VectorXd::Zero(d_out);
        for (int i = 0; i < k_active; ++i) {
            int idx = indices[i];
            double weight = std::exp(gate_logits(idx)); // Simplified softmax
            output += weight * (experts[idx] * h);
        }
        return output;
    }

private:
    int d_in;
    int d_out;
    int n_experts;
    int k_active;
    std::vector<Eigen::MatrixXd> experts;
    Eigen::MatrixXd W_gate;
};

#endif // SSOG_HPP
