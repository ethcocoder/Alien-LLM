#ifndef ORCHESTRATOR_HPP
#define ORCHESTRATOR_HPP

#include <vector>
#include <fstream>
#include <iostream>
#include "tokenizer.hpp"
#include "embeddings.hpp"
#include "ssm.hpp"
#include "rfa.hpp"
#include "stre.hpp"
#include "uq.hpp"
#include "ata.hpp"
#include "ssog.hpp"

class AI2Orchestrator {
public:
    AI2Orchestrator(int vocab_size, int d_model) 
        : embedding(vocab_size, d_model),
          ssm(d_model, 32),     
          rfa(d_model, 16),     
          stre(d_model, 32),    
          ata(16, d_model),
          ssog(32, d_model, 8, 2) { 
        this->d_model = d_model;
        this->vocab_size = vocab_size;


        
        // Output Layer (The "Brain")
        W_out = Eigen::MatrixXf::Random(vocab_size, d_model) * 0.01f;
        W_out_acc = Eigen::MatrixXf::Zero(vocab_size, d_model);

        
        // Normalization params
        gamma = Eigen::VectorXf::Ones(d_model);
        beta = Eigen::VectorXf::Zero(d_model);
        last_h_reason = Eigen::VectorXf::Zero(d_model);
    }

    Eigen::VectorXf process_token(int token_id, const Eigen::VectorXf& task_emb) {
        Eigen::VectorXf x = embedding.get_embedding(token_id);
        
        // ATA Adaptation
        Eigen::MatrixXf adapter = ata.generate_adapter(task_emb);
        x = x + adapter * x;

        // Core processing
        Eigen::VectorXf h_ssm = ssm.forward(x);
        Eigen::VectorXf h_rfa = rfa.forward(x, x, x);
        
        // Gating & Normalization
        Eigen::VectorXf h = 0.5f * h_ssm + 0.5f * h_rfa;
        h = layer_norm(h);

        // Reasoning (Bottleneck to 32 dimensions)
        history.push_back(h);
        if (history.size() > 5) history.erase(history.begin());
        Eigen::VectorXf h_reason = stre.forward(history);

        // Synthesis (Expand back to d_model=256 dimensions)
        Eigen::VectorXf h_final = ssog.forward(h_reason);
        this->last_h_reason = h_final; // Renamed logic: this is actually the final hidden state before W_out
        
        Eigen::VectorXf logits = W_out * h_final;
        return logits;
    }



    Eigen::VectorXf layer_norm(const Eigen::VectorXf& x) {
        float mean = x.mean();
        float var = (x.array() - mean).square().mean();
        return gamma.array() * ((x.array() - mean) / std::sqrt(var + 1e-6f)) + beta.array();
    }

    void reset() {
        ssm.reset_state();
        rfa.reset_state();
        history.clear();
    }

    void accumulate_gradients(int token_id, const Eigen::VectorXf& task_emb, const Eigen::VectorXf& grad_logits) {
        // Accumulate W_out
        W_out_acc += grad_logits * last_h_reason.transpose();

        // Propagate to layers
        Eigen::VectorXf grad_h = W_out.transpose() * grad_logits;
        
        Eigen::VectorXf x = embedding.get_embedding(token_id);
        ssm.accumulate_gradients(x, 0.5f * grad_h);
        rfa.accumulate_gradients(x, 0.5f * grad_h);
    }

    void apply_gradients(float lr) {
        // Update W_out with Adam logic
        static Eigen::MatrixXf m_W = Eigen::MatrixXf::Zero(vocab_size, d_model);
        static Eigen::MatrixXf v_W = Eigen::MatrixXf::Zero(vocab_size, d_model);
        static int t = 0;
        t++;
        
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        m_W = b1 * m_W + (1.0f - b1) * W_out_acc;
        v_W = b2 * v_W + (1.0f - b2) * W_out_acc.array().square().matrix();
        
        float m_corr = 1.0f - std::pow(b1, t);
        float v_corr = 1.0f - std::pow(b2, t);
        W_out.array() -= lr * (m_W.array() / m_corr) / ((v_W.array() / v_corr).sqrt() + eps);
        
        W_out_acc.setZero();
        
        // Apply to sub-layers
        ssm.apply_gradients(lr);
        rfa.apply_gradients(lr);
    }



    int get_d_model() const { return d_model; }
    int get_vocab_size() const { return vocab_size; }

    void save_checkpoint(const std::string& path) const {
        std::ofstream os(path, std::ios::binary);
        if (!os.is_open()) return;

        // Header: Global Dimensions
        os.write((char*)&vocab_size, sizeof(int));
        os.write((char*)&d_model, sizeof(int));

        embedding.save(os);
        ssm.save(os);
        rfa.save(os);
        stre.save(os);
        ata.save(os);
        ssog.save(os);
        uq.save(os);
        
        // Save Brain weights
        os.write((char*)W_out.data(), W_out.size() * sizeof(float));
        os.write((char*)gamma.data(), gamma.size() * sizeof(float));
        os.write((char*)beta.data(), beta.size() * sizeof(float));
    }

    void load_checkpoint(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        if (is.is_open()) {
            is.read((char*)&vocab_size, sizeof(int));
            is.read((char*)&d_model, sizeof(int));

            embedding.load(is);
            ssm.load(is);
            rfa.load(is);
            stre.load(is);
            ata.load(is);
            ssog.load(is);
            uq.load(is);
            
            is.read((char*)W_out.data(), W_out.size() * sizeof(float));
            is.read((char*)gamma.data(), gamma.size() * sizeof(float));
            is.read((char*)beta.data(), beta.size() * sizeof(float));
        }
    }


private:
    int d_model;
    int vocab_size;
    CHEEmbedding embedding;
    S4DLayer ssm;
    RFALayer rfa;
    SheafLayer stre;
    UncertaintyQuantifier uq;
    AgilityMetaController ata;
    SparseSynthesizer ssog;
    
    // Brain (Output)
    Eigen::MatrixXf W_out, W_out_acc;
    Eigen::VectorXf gamma, beta;
    Eigen::VectorXf last_h_reason;
    
    std::vector<Eigen::VectorXf> history;
};




#endif // ORCHESTRATOR_HPP

