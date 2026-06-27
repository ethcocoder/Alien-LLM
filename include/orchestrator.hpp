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
    }

    Eigen::VectorXf process_token(int token_id, const Eigen::VectorXf& task_emb) {
        Eigen::VectorXf x = embedding.get_embedding(token_id);
        
        // ATA Adaptation
        Eigen::MatrixXf adapter = ata.generate_adapter(task_emb);
        x = x + adapter * x;

        // Core processing
        Eigen::VectorXf h_ssm = ssm.forward(x);
        Eigen::VectorXf h_rfa = rfa.forward(x, x, x);
        
        // Gating (simplified)
        Eigen::VectorXf h = 0.5f * h_ssm + 0.5f * h_rfa;

        // Reasoning (using a small window of history for demonstration)
        history.push_back(h);
        if (history.size() > 5) history.erase(history.begin());
        Eigen::VectorXf h_reason = stre.forward(history);

        // Synthesis
        Eigen::VectorXf out = ssog.forward(h_reason);
        return out;
    }

    void reset() {
        ssm.reset_state();
        rfa.reset_state();
        history.clear();
    }

    int get_d_model() const { return d_model; }

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
            is.close();
        } else {
            std::cerr << "[ERROR] Could not open " << path << " for reading." << std::endl;
        }
    }

private:
    int d_model;
    CHEEmbedding embedding;
    S4DLayer ssm;
    RFALayer rfa;
    SheafLayer stre;
    UncertaintyQuantifier uq;
    AgilityMetaController ata;
    SparseSynthesizer ssog;
    
    std::vector<Eigen::VectorXf> history;
};

#endif // ORCHESTRATOR_HPP

