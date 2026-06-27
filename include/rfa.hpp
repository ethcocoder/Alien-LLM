#ifndef RFA_HPP
#define RFA_HPP

#include <vector>
#include <cmath>
#include <Eigen/Dense>

class RFALayer {
public:
    RFALayer(int d_model, int r_features) : d_model(d_model), r_features(r_features) {
        W_random = Eigen::MatrixXf::Random(r_features, d_model);
        
        // Adam parameters
        m_W = Eigen::MatrixXf::Zero(r_features, d_model);
        v_W = Eigen::MatrixXf::Zero(r_features, d_model);
        grad_W_acc = Eigen::MatrixXf::Zero(r_features, d_model);
        t = 0;
        
        // Initialize state for prefix sums (causal attention)

        numerator_state = Eigen::MatrixXf::Zero(r_features, d_model);
        denominator_state = Eigen::VectorXf::Zero(r_features);
    }

    Eigen::VectorXf forward(const Eigen::VectorXf& q, const Eigen::VectorXf& k, const Eigen::VectorXf& v) {
        Eigen::VectorXf phi_q = feature_map(q);
        Eigen::VectorXf phi_k = feature_map(k);
        
        // Update states (prefix sums)
        numerator_state += phi_k * v.transpose();
        denominator_state += phi_k;
        
        // Compute attention output
        Eigen::VectorXf num = numerator_state.transpose() * phi_q;
        float den = denominator_state.dot(phi_q) + 1e-6f;
        
        return num / den;
    }

    void reset_state() {
        numerator_state.setZero();
        denominator_state.setZero();
    }

    void save(std::ostream& os) const {
        os.write((char*)W_random.data(), W_random.size() * sizeof(float));
    }

    void load(std::istream& is) {
        is.read((char*)W_random.data(), W_random.size() * sizeof(float));
    }

    void accumulate_gradients(const Eigen::VectorXf& q, const Eigen::VectorXf& grad_out) {
        Eigen::VectorXf phi_q = feature_map(q);
        Eigen::VectorXf grad_phi = (W_random * grad_out).cwiseProduct(phi_q);
        grad_W_acc += grad_phi * q.transpose();
    }

    void apply_gradients(float lr) {
        t++;
        // Adam Update
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        m_W = b1 * m_W + (1.0f - b1) * grad_W_acc;
        v_W = b2 * v_W + (1.0f - b2) * grad_W_acc.array().square().matrix();
        
        float m_corr = 1.0f - std::pow(b1, t);
        float v_corr = 1.0f - std::pow(b2, t);
        
        W_random.array() -= lr * (m_W.array() / m_corr) / ((v_W.array() / v_corr).sqrt() + eps);
        
        // Reset
        grad_W_acc.setZero();
    }


private:


    int d_model;
    int r_features;
    Eigen::MatrixXf W_random;
    Eigen::MatrixXf m_W, v_W;
    Eigen::MatrixXf grad_W_acc;
    int t;
    
    Eigen::MatrixXf numerator_state;


    Eigen::VectorXf denominator_state;

    Eigen::VectorXf feature_map(const Eigen::VectorXf& x) {
        // Performer's random feature map: phi(x) = exp(w^T x - ||x||^2/2)
        Eigen::VectorXf projection = W_random * x;
        float norm_sq = x.squaredNorm();
        return (projection.array() - norm_sq / 2.0f).exp();
    }
};

#endif // RFA_HPP

