#ifndef EMBEDDINGS_HPP
#define EMBEDDINGS_HPP

#include <vector>
#include <Eigen/Dense>
#include <functional>

class CHEEmbedding {
public:
    CHEEmbedding(int vocab_size, int d_model, int k_shards = 4, int m_hash = 65536)
        : d_model(d_model), k_shards(k_shards), m_hash(m_hash) {
        
        shard_dim = d_model / k_shards;
        for (int i = 0; i < k_shards; ++i) {
            shards.push_back(Eigen::MatrixXd::Random(m_hash, shard_dim));
        }
    }

    Eigen::VectorXd get_embedding(int token_id) {
        Eigen::VectorXd result(d_model);
        for (int j = 0; j < k_shards; ++j) {
            int hash_val = hash_func(token_id, j) % m_hash;
            result.segment(j * shard_dim, shard_dim) = shards[j].row(hash_val);
        }
        return result;
    }

private:
    int d_model;
    int k_shards;
    int m_hash;
    int shard_dim;
    std::vector<Eigen::MatrixXd> shards;

    size_t hash_func(int token_id, int shard_idx) {
        // Simple universal hash function variant
        size_t h = std::hash<int>{}(token_id);
        h ^= std::hash<int>{}(shard_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

#endif // EMBEDDINGS_HPP
