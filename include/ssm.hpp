#ifndef SSM_HPP
#define SSM_HPP

#include <vector>
#include <complex>
#include <Eigen/Dense>

class S4DLayer {
public:
    S4DLayer(int d_model, int d_state) : d_model(d_model), d_state(d_state) {
        // Initialize SSM parameters
        Lambda = Eigen::VectorXcd::Random(d_state);
        // Ensure stability: real part should be negative for continuous, 
        // or magnitude < 1 for discrete. SPEC says lambda = e^(omega + i*phi)
        for(int i=0; i<d_state; ++i) {
            Lambda(i) = std::complex<double>(-0.1, 1.0) * ((double)rand() / (double)RAND_MAX);
        }
        
        B = Eigen::MatrixXcd::Random(d_state, d_model);
        C = Eigen::MatrixXcd::Random(d_model, d_state);
        D = Eigen::VectorXd::Random(d_model);
        
        state = Eigen::VectorXcd::Zero(d_state);
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& u) {
        // x_{k+1} = Lambda * x_k + B * u_k
        // y_k = Re(C * x_k) + D * u_k
        
        // Element-wise multiplication for diagonal Lambda
        state = Lambda.cwiseProduct(state) + B * u.cast<std::complex<double>>();
        
        Eigen::VectorXcd output_complex = C * state;
        Eigen::VectorXd y = output_complex.real() + D.cwiseProduct(u);
        
        return y;
    }

    void reset_state() {
        state = Eigen::VectorXcd::Zero(d_state);
    }

    void save(std::ostream& os) const {
        os.write((char*)&d_model, sizeof(d_model));
        os.write((char*)&d_state, sizeof(d_state));
        os.write((char*)Lambda.data(), Lambda.size() * sizeof(std::complex<double>));
        os.write((char*)B.data(), B.size() * sizeof(std::complex<double>));
        os.write((char*)C.data(), C.size() * sizeof(std::complex<double>));
        os.write((char*)D.data(), D.size() * sizeof(double));
    }

    void load(std::istream& is) {
        int file_d_model, file_d_state;
        is.read((char*)&file_d_model, sizeof(file_d_model));
        is.read((char*)&file_d_state, sizeof(file_d_state));
        
        if (file_d_model != d_model || file_d_state != d_state) {
            std::cerr << "[ERROR] S4DLayer: Mismatched architecture in checkpoint. "
                      << "File: " << file_d_model << "/" << file_d_state << ", "
                      << "Model: " << d_model << "/" << d_state << std::endl;
            // Skip reading data to avoid corruption if sizes differ
            is.seekg(Lambda.size() * sizeof(std::complex<double>) + 
                     B.size() * sizeof(std::complex<double>) + 
                     C.size() * sizeof(std::complex<double>) + 
                     D.size() * sizeof(double), std::ios::cur);
            return;
        }

        is.read((char*)Lambda.data(), Lambda.size() * sizeof(std::complex<double>));
        is.read((char*)B.data(), B.size() * sizeof(std::complex<double>));
        is.read((char*)C.data(), C.size() * sizeof(std::complex<double>));
        is.read((char*)D.data(), D.size() * sizeof(double));
    }

private:
    int d_model;
    int d_state;
    Eigen::VectorXcd Lambda;
    Eigen::MatrixXcd B;
    Eigen::MatrixXcd C;
    Eigen::VectorXd D;
    Eigen::VectorXcd state;
};

#endif // SSM_HPP
