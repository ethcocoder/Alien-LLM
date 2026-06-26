#ifndef ORCHESTRATOR_HPP
#define ORCHESTRATOR_HPP

#include <vector>
#include <fstream>
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

    Eigen::VectorXd process_token(int token_id, const Eigen::VectorXd& task_emb) {
        Eigen::VectorXd x = embedding.get_embedding(token_id);
        
        // ATA Adaptation
        Eigen::MatrixXd adapter = ata.generate_adapter(task_emb);
        x = x + adapter * x;

        // Core processing
        Eigen::VectorXd h_ssm = ssm.forward(x);
        Eigen::VectorXd h_rfa = rfa.forward(x, x, x);
        
        // Gating (simplified)
        Eigen::VectorXd h = 0.5 * h_ssm + 0.5 * h_rfa;

        // Reasoning (using a small window of history for demonstration)
        history.push_back(h);
        if (history.size() > 5) history.erase(history.begin());
        Eigen::VectorXd h_reason = stre.forward(history);

        // Synthesis
        Eigen::VectorXd out = ssog.forward(h_reason);
        return out;
    }

    void reset() {
        ssm.reset_state();
        rfa.reset_state();
        history.clear();
    }

    void save_checkpoint(const std::string& path) const {
        std::ofstream os(path, std::ios::binary);
        embedding.save(os);
        ssm.save(os);
        // Add other components if needed
        os.close();
    }

    void load_checkpoint(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        if (is.is_open()) {
            embedding = CHEEmbedding::load(is);
            ssm.load(is);
            is.close();
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
    
    std::vector<Eigen::VectorXd> history;
};

#endif // ORCHESTRATOR_HPP
