#ifndef SSOG_HPP
#define SSOG_HPP

#include <vector>
#include <Eigen/Dense>
#include <numeric>

class SparseSynthesizer {
public:
    SparseSynthesizer(int d_in, int d_out, int n_experts, int k_active) 
        : d_in(d_in), d_out(d_out), n_experts(n_experts), k_active(k_active) {
        for (int i = 0; i < n_experts; ++i) {
            experts.push_back(Eigen::MatrixXf::Random(d_out, d_in));
        }
        W_gate = Eigen::MatrixXf::Random(n_experts, d_in);
    }

    Eigen::VectorXf forward(const Eigen::VectorXf& h) {
        Eigen::VectorXf gate_logits = W_gate * h;
        
        // Simple top-k routing
        std::vector<int> indices(n_experts);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k_active, indices.end(),
                          [&](int i, int j) { return gate_logits(i) > gate_logits(j); });

        Eigen::VectorXf output = Eigen::VectorXf::Zero(d_out);
        for (int i = 0; i < k_active; ++i) {
            int idx = indices[i];
            float weight = std::exp(gate_logits(idx)); // Simplified softmax
            output += weight * (experts[idx] * h);
        }
        return output;
    }

    void save(std::ostream& os) const {
        for (const auto& expert : experts) {
            os.write((char*)expert.data(), expert.size() * sizeof(float));
        }
        os.write((char*)W_gate.data(), W_gate.size() * sizeof(float));
    }

    void load(std::istream& is) {
        for (auto& expert : experts) {
            is.read((char*)expert.data(), expert.size() * sizeof(float));
        }
        is.read((char*)W_gate.data(), W_gate.size() * sizeof(float));
    }

private:
    int d_in;
    int d_out;
    int n_experts;
    int k_active;
    std::vector<Eigen::MatrixXf> experts;
    Eigen::MatrixXf W_gate;
};

#endif // SSOG_HPP

