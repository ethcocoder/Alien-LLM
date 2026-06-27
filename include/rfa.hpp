#ifndef RFA_HPP
#define RFA_HPP

#include <vector>
#include <cmath>
#include <Eigen/Dense>

class RFALayer {
public:
    RFALayer(int d_model, int r_features) : d_model(d_model), r_features(r_features) {
        W_random = Eigen::MatrixXd::Random(r_features, d_model);
        
        // Initialize state for prefix sums (causal attention)
        numerator_state = Eigen::MatrixXd::Zero(r_features, d_model);
        denominator_state = Eigen::VectorXd::Zero(r_features);
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& q, const Eigen::VectorXd& k, const Eigen::VectorXd& v) {
        Eigen::VectorXd phi_q = feature_map(q);
        Eigen::VectorXd phi_k = feature_map(k);
        
        // Update states (prefix sums)
        numerator_state += phi_k * v.transpose();
        denominator_state += phi_k;
        
        // Compute attention output
        Eigen::VectorXd num = numerator_state.transpose() * phi_q;
        double den = denominator_state.dot(phi_q) + 1e-6;
        
        return num / den;
    }

    void reset_state() {
        numerator_state.setZero();
        denominator_state.setZero();
    }

    void save(std::ostream& os) const {
        os.write((char*)W_random.data(), W_random.size() * sizeof(double));
    }

    void load(std::istream& is) {
        is.read((char*)W_random.data(), W_random.size() * sizeof(double));
    }

private:
    int d_model;
    int r_features;
    Eigen::MatrixXd W_random;
    
    Eigen::MatrixXd numerator_state;
    Eigen::VectorXd denominator_state;

    Eigen::VectorXd feature_map(const Eigen::VectorXd& x) {
        // Performer's random feature map: phi(x) = exp(w^T x - ||x||^2/2)
        Eigen::VectorXd projection = W_random * x;
        double norm_sq = x.squaredNorm();
        return (projection.array() - norm_sq / 2.0).exp();
    }
};

#endif // RFA_HPP
