#ifndef UQ_HPP
#define UQ_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

class UncertaintyQuantifier {
public:
    UncertaintyQuantifier(int reservoir_size = 100) : reservoir_size(reservoir_size) {}

    void update_reservoir(float score) {
        if (scores.size() < reservoir_size) {
            scores.push_back(score);
        } else {
            scores[rand() % reservoir_size] = score;
        }
    }

    float get_conformal_threshold(float alpha = 0.1f) {
        if (scores.empty()) return 1.0f;
        std::vector<float> sorted_scores = scores;
        std::sort(sorted_scores.begin(), sorted_scores.end());
        int idx = static_cast<int>((1.0f - alpha) * sorted_scores.size());
        return sorted_scores[std::min(idx, (int)sorted_scores.size() - 1)];
    }

    float compute_total_uncertainty(const std::vector<Eigen::VectorXf>& ensemble_outputs) {
        if (ensemble_outputs.empty()) return 0.0f;
        
        Eigen::VectorXf mean = Eigen::VectorXf::Zero(ensemble_outputs[0].size());
        for (const auto& out : ensemble_outputs) mean += out;
        mean /= (float)ensemble_outputs.size();

        float variance = 0.0f;
        for (const auto& out : ensemble_outputs) {
            variance += (out - mean).squaredNorm();
        }
        return variance / (float)ensemble_outputs.size();
    }

    void save(std::ostream& os) const {
        size_t size = scores.size();
        os.write((char*)&size, sizeof(size));
        if (size > 0) {
            os.write((char*)scores.data(), size * sizeof(float));
        }
    }

    void load(std::istream& is) {
        size_t size;
        is.read((char*)&size, sizeof(size));
        scores.resize(size);
        if (size > 0) {
            is.read((char*)scores.data(), size * sizeof(float));
        }
    }

private:
    int reservoir_size;
    std::vector<float> scores;
};

#endif // UQ_HPP

