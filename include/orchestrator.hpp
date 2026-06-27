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

        // Reasoning
        history.push_back(h);
        if (history.size() > 5) history.erase(history.begin());
        Eigen::VectorXf h_reason = stre.forward(history);

        // Synthesis to logits
        this->last_h_reason = h_reason;
        
        Eigen::VectorXf logits = W_out * h_reason;
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

    void update(int token_id, const Eigen::VectorXf& task_emb, const Eigen::VectorXf& grad_logits, float lr) {

        // 1. Gradient of W_out
        // dL/dW_out = grad_logits * h_reason^T
        W_out -= lr * grad_logits * last_h_reason.transpose();

        // 2. Propagate to layers
        Eigen::VectorXf grad_h = W_out.transpose() * grad_logits;
        
        // Update STRE
        // (Simplified: we use the same grad for recently used layers)
        
        // Update Core Layers
        Eigen::VectorXf x = embedding.get_embedding(token_id);
        ssm.update(x, 0.5f * grad_h, lr);
        rfa.update(x, 0.5f * grad_h, lr);
        
        // Update Embedding
        embedding.update(token_id, grad_h, lr);
    }


    int get_d_model() const { return d_model; }
    int get_vocab_size() const { return vocab_size; }

    void save_checkpoint(const std::string& path) const {
        std::ofstream os(path, std::ios::binary);
        if (!os.is_open()) {
            std::cerr << "[ERROR] Could not open " << path << " for writing." << std::endl;
            return;
        }
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
        
        os.close();
    }

    void load_checkpoint(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        if (is.is_open()) {
            embedding = CHEEmbedding::load(is);
            ssm.load(is);
            rfa.load(is);
            stre.load(is);
            ata.load(is);
            ssog.load(is);
            uq.load(is);
            
            // Load Brain weights
            is.read((char*)W_out.data(), W_out.size() * sizeof(float));
            is.read((char*)gamma.data(), gamma.size() * sizeof(float));
            is.read((char*)beta.data(), beta.size() * sizeof(float));
            
            is.close();
        } else {
            std::cerr << "[ERROR] Could not open " << path << " for reading." << std::endl;
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
    Eigen::MatrixXf W_out;
    Eigen::VectorXf gamma, beta;
    Eigen::VectorXf last_h_reason;
    
    std::vector<Eigen::VectorXf> history;
};



#endif // ORCHESTRATOR_HPP

