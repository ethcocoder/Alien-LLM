#ifndef UQ_HPP
#define UQ_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

class UncertaintyQuantifier {
public:
    UncertaintyQuantifier(int reservoir_size = 100) : reservoir_size(reservoir_size) {}

    void update_reservoir(double score) {
        if (scores.size() < reservoir_size) {
            scores.push_back(score);
        } else {
            scores[rand() % reservoir_size] = score;
        }
    }

    double get_conformal_threshold(double alpha = 0.1) {
        if (scores.empty()) return 1.0;
        std::vector<double> sorted_scores = scores;
        std::sort(sorted_scores.begin(), sorted_scores.end());
        int idx = static_cast<int>((1.0 - alpha) * sorted_scores.size());
        return sorted_scores[std::min(idx, (int)sorted_scores.size() - 1)];
    }

    double compute_total_uncertainty(const std::vector<Eigen::VectorXd>& ensemble_outputs) {
        if (ensemble_outputs.empty()) return 0.0;
        
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(ensemble_outputs[0].size());
        for (const auto& out : ensemble_outputs) mean += out;
        mean /= ensemble_outputs.size();

        double variance = 0.0;
        for (const auto& out : ensemble_outputs) {
            variance += (out - mean).squaredNorm();
        }
        return variance / ensemble_outputs.size();
    }

private:
    int reservoir_size;
    std::vector<double> scores;
};

#endif // UQ_HPP
