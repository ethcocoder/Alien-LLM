#ifndef ATA_HPP
#define ATA_HPP

#include <Eigen/Dense>

class AgilityMetaController {
public:
    AgilityMetaController(int d_task, int d_model) : d_task(d_task), d_model(d_model) {
        W_hyper = Eigen::MatrixXf::Random(d_model * d_model, d_task);
    }

    Eigen::MatrixXf generate_adapter(const Eigen::VectorXf& task_embedding) {
        Eigen::VectorXf flat_weights = W_hyper * task_embedding;
        Eigen::MatrixXf adapter = Eigen::Map<Eigen::MatrixXf>(flat_weights.data(), d_model, d_model);
        return adapter;
    }

    void save(std::ostream& os) const {
        os.write((char*)W_hyper.data(), W_hyper.size() * sizeof(float));
    }

    void load(std::istream& is) {
        is.read((char*)W_hyper.data(), W_hyper.size() * sizeof(float));
    }

private:
    int d_task;
    int d_model;
    Eigen::MatrixXf W_hyper;
};

#endif // ATA_HPP

