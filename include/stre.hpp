#ifndef STRE_HPP
#define STRE_HPP

#include <vector>
#include <Eigen/Dense>

class SheafLayer {
public:
    SheafLayer(int d_model, int d_reasoning) : d_model(d_model), d_reasoning(d_reasoning) {
        // Restriction maps for a simple graph (e.g., a cycle or path)
        W_restriction = Eigen::MatrixXf::Random(d_reasoning, d_reasoning);
        W_node = Eigen::MatrixXf::Random(d_reasoning, d_model);
    }

    Eigen::VectorXf forward(const std::vector<Eigen::VectorXf>& node_features) {
        int n = node_features.size();
        std::vector<Eigen::VectorXf> x(n);
        for(int i=0; i<n; ++i) {
            x[i] = W_node * node_features[i];
        }

        std::vector<Eigen::VectorXf> next_x(n, Eigen::VectorXf::Zero(d_reasoning));
        for(int i=0; i<n; ++i) {
            // Simple propagation to neighbors (i-1, i+1)
            int prev = (i - 1 + n) % n;
            int next = (i + 1) % n;
            
            next_x[i] = x[i] + W_restriction * x[prev] + W_restriction * x[next];
            next_x[i] = next_x[i].array().tanh(); // Activation
        }

        // Aggregate or return the last node's reasoning state
        return next_x.back();
    }

    void save(std::ostream& os) const {
        os.write((char*)W_restriction.data(), W_restriction.size() * sizeof(float));
        os.write((char*)W_node.data(), W_node.size() * sizeof(float));
    }

    void load(std::istream& is) {
        is.read((char*)W_restriction.data(), W_restriction.size() * sizeof(float));
        is.read((char*)W_node.data(), W_node.size() * sizeof(float));
    }

private:
    int d_model;
    int d_reasoning;
    Eigen::MatrixXf W_restriction;
    Eigen::MatrixXf W_node;
};

#endif // STRE_HPP

