#ifndef SSM_HPP
#define SSM_HPP

#include <vector>
#include <complex>
#include <Eigen/Dense>

class S4DLayer {
public:
    S4DLayer(int d_model, int d_state) : d_model(d_model), d_state(d_state) {
        // Initialize SSM parameters
        Lambda = Eigen::VectorXcf::Random(d_state);
        // Ensure stability: real part should be negative for continuous, 
        // or magnitude < 1 for discrete. SPEC says lambda = e^(omega + i*phi)
        for(int i=0; i<d_state; ++i) {
            Lambda(i) = std::complex<float>(-0.1f, 1.0f) * ((float)rand() / (float)RAND_MAX);
        }
        
        B = Eigen::MatrixXcf::Random(d_state, d_model);
        C = Eigen::MatrixXcf::Random(d_model, d_state);
        D = Eigen::VectorXf::Random(d_model);
        
        state = Eigen::VectorXcf::Zero(d_state);
        
        // Adam parameters
        m_C = Eigen::MatrixXcf::Zero(d_model, d_state);
        v_C = Eigen::MatrixXf::Zero(d_model, d_state);
        t = 0;
    }


    Eigen::VectorXf forward(const Eigen::VectorXf& u) {
        // x_{k+1} = Lambda * x_k + B * u_k
        // y_k = Re(C * x_k) + D * u_k
        
        // Element-wise multiplication for diagonal Lambda
        state = Lambda.cwiseProduct(state) + B * u.cast<std::complex<float>>();
        
        Eigen::VectorXcf output_complex = C * state;
        Eigen::VectorXf y = output_complex.real() + D.cwiseProduct(u);
        
        return y;
    }

    void reset_state() {
        state = Eigen::VectorXcf::Zero(d_state);
    }

    void save(std::ostream& os) const {
        os.write((char*)&d_model, sizeof(d_model));
        os.write((char*)&d_state, sizeof(d_state));
        os.write((char*)Lambda.data(), Lambda.size() * sizeof(std::complex<float>));
        os.write((char*)B.data(), B.size() * sizeof(std::complex<float>));
        os.write((char*)C.data(), C.size() * sizeof(std::complex<float>));
        os.write((char*)D.data(), D.size() * sizeof(float));
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
            is.seekg(Lambda.size() * sizeof(std::complex<float>) + 
                     B.size() * sizeof(std::complex<float>) + 
                     C.size() * sizeof(std::complex<float>) + 
                     D.size() * sizeof(float), std::ios::cur);
            return;
        }

        is.read((char*)Lambda.data(), Lambda.size() * sizeof(std::complex<float>));
        is.read((char*)B.data(), B.size() * sizeof(std::complex<float>));
        is.read((char*)C.data(), C.size() * sizeof(std::complex<float>));
        is.read((char*)D.data(), D.size() * sizeof(float));
    }

    void update(const Eigen::VectorXf& u, const Eigen::VectorXf& grad_out, float lr = 0.001f) {
        t++;
        Eigen::MatrixXcf grad_C = grad_out.cast<std::complex<float>>() * state.adjoint();
        
        // Adam Update for C
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        m_C = b1 * m_C + (1.0f - b1) * grad_C;
        v_C = b2 * v_C + (1.0f - b2) * grad_C.array().abs2().matrix();
        
        Eigen::MatrixXcf m_hat = m_C / (1.0f - std::pow(b1, t));
        Eigen::MatrixXf v_hat = v_C / (1.0f - std::pow(b2, t));
        
        for(int i=0; i<C.rows(); ++i) {
            for(int j=0; j<C.cols(); ++j) {
                C(i,j) -= lr * m_hat(i,j) / (std::sqrt(v_hat(i,j)) + eps);
            }
        }

        // Simpler update for others
        B -= lr * (state * u.cast<std::complex<float>>().transpose().conjugate()).topRows(d_state);
        D -= lr * grad_out.cwiseProduct(u);
    }


private:

    int d_model;
    int d_state;
    Eigen::VectorXcf Lambda;
    Eigen::MatrixXcf B;
    Eigen::MatrixXcf C;
    Eigen::VectorXf D;
    Eigen::VectorXcf state;
    
    // Adam State
    Eigen::MatrixXcf m_C;
    Eigen::MatrixXf v_C;
    int t;
};


#endif // SSM_HPP

