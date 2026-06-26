#ifndef ATA_HPP
#define ATA_HPP

#include <Eigen/Dense>

class AgilityMetaController {
public:
    AgilityMetaController(int d_task, int d_model) : d_task(d_task), d_model(d_model) {
        W_hyper = Eigen::MatrixXd::Random(d_model * d_model, d_task);
    }

    Eigen::MatrixXd generate_adapter(const Eigen::VectorXd& task_embedding) {
        Eigen::VectorXd flat_weights = W_hyper * task_embedding;
        Eigen::MatrixXd adapter = Eigen::Map<Eigen::MatrixXd>(flat_weights.data(), d_model, d_model);
        return adapter;
    }

private:
    int d_task;
    int d_model;
    Eigen::MatrixXd W_hyper;
};

#endif // ATA_HPP
