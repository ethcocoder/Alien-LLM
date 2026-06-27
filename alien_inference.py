import numpy as np
import os
import struct

class AlienInference:
    def __init__(self, vocab_path, model_path):
        self.load_vocab(vocab_path)
        self.load_model(model_path)
        self.history_nodes = []

    def load_vocab(self, path):
        self.vocab = {}
        self.id_to_token = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    idx, token = int(parts[0]), parts[1]
                    self.vocab[token] = idx
                    self.id_to_token[idx] = token
        self.vocab_size = len(self.vocab)

    def load_model(self, path):
        with open(path, 'rb') as f:
            # ============================================================
            # 0. Global Header (written by orchestrator)
            # ============================================================
            self.vocab_size = struct.unpack('i', f.read(4))[0]
            self.d_model = struct.unpack('i', f.read(4))[0]

            # ============================================================
            # 1. Embedding (CHEEmbedding::save)
            #    Writes: d_model(int), k_shards(int), m_hash(int), shard_dim(int)
            #    Then: k_shards matrices of shape (m_hash, shard_dim) in column-major
            # ============================================================
            self.emb_d_model = struct.unpack('i', f.read(4))[0]
            self.emb_k_shards = struct.unpack('i', f.read(4))[0]
            self.emb_m_hash = struct.unpack('i', f.read(4))[0]
            self.emb_shard_dim = struct.unpack('i', f.read(4))[0]

            self.shards = []
            for _ in range(self.emb_k_shards):
                n_floats = self.emb_m_hash * self.emb_shard_dim
                shard_data = np.frombuffer(f.read(n_floats * 4), dtype=np.float32)
                self.shards.append(shard_data.reshape(self.emb_m_hash, self.emb_shard_dim, order='F'))

            # ============================================================
            # 2. SSM / S4DLayer::save
            #    Writes: d_model(int), d_state(int)  <-- HEADER!
            #    Then: Lambda(complex, d_state), B(complex, d_state x d_model),
            #          C(complex, d_model x d_state), D(float, d_model)
            # ============================================================
            ssm_d_model = struct.unpack('i', f.read(4))[0]
            ssm_d_state = struct.unpack('i', f.read(4))[0]
            self.d_state = ssm_d_state

            # Lambda: d_state complex numbers = d_state * 2 floats
            lambda_data = np.frombuffer(f.read(self.d_state * 2 * 4), dtype=np.float32)
            self.ssm_lambda = lambda_data[::2] + 1j * lambda_data[1::2]

            # B: (d_state, d_model) complex = d_state * d_model * 2 floats
            n = self.d_state * self.d_model * 2
            self.ssm_B = np.frombuffer(f.read(n * 4), dtype=np.float32).view(np.complex64).reshape(self.d_state, self.d_model, order='F')

            # C: (d_model, d_state) complex
            n = self.d_model * self.d_state * 2
            self.ssm_C = np.frombuffer(f.read(n * 4), dtype=np.float32).view(np.complex64).reshape(self.d_model, self.d_state, order='F')

            # D: (d_model,) real
            self.ssm_D = np.frombuffer(f.read(self.d_model * 4), dtype=np.float32)
            self.ssm_state = np.zeros(self.d_state, dtype=np.complex64)

            # ============================================================
            # 3. RFA (RFALayer::save)
            #    Writes: W_random (r_features x d_model) floats, column-major
            #    r_features = 16 (from orchestrator constructor)
            # ============================================================
            self.rfa_features = 16
            n = self.rfa_features * self.d_model
            self.rfa_W = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(self.rfa_features, self.d_model, order='F')
            self.rfa_num = np.zeros((self.rfa_features, self.d_model), dtype=np.float32)
            self.rfa_den = np.zeros(self.rfa_features, dtype=np.float32)

            # ============================================================
            # 4. STRE / SheafLayer::save
            #    Writes: W_restriction (d_reasoning x d_reasoning), W_node (d_reasoning x d_model)
            #    d_reasoning = 32 (from orchestrator constructor)
            # ============================================================
            d_reasoning = self.d_state  # Both are 32
            n = d_reasoning * d_reasoning
            self.stre_W_res = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(d_reasoning, d_reasoning, order='F')
            n = d_reasoning * self.d_model
            self.stre_W_node = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(d_reasoning, self.d_model, order='F')

            # ============================================================
            # 5. ATA / AgilityMetaController::save
            #    Writes: W_hyper ((d_model*d_model) x d_task) floats, column-major
            #    d_task = 16 (from orchestrator constructor)
            # ============================================================
            d_task = 16
            n = self.d_model * self.d_model * d_task
            self.ata_W = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(self.d_model * self.d_model, d_task, order='F')

            # ============================================================
            # 6. SSOG / SparseSynthesizer::save
            #    Writes: n_experts matrices of (d_out x d_in), then W_gate (n_experts x d_in)
            #    d_in = 32, d_out = 256 (d_model), n_experts = 8
            # ============================================================
            ssog_d_in = 32   # from orchestrator: ssog(32, d_model, 8, 2) -> d_in=32
            ssog_d_out = self.d_model
            n_experts = 8
            self.experts = []
            for _ in range(n_experts):
                n = ssog_d_out * ssog_d_in
                expert = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(ssog_d_out, ssog_d_in, order='F')
                self.experts.append(expert)
            n = n_experts * ssog_d_in
            self.ssog_W_gate = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(n_experts, ssog_d_in, order='F')

            # ============================================================
            # 7. UQ / UncertaintyQuantifier::save
            #    Writes: size_t (8 bytes on 64-bit Linux!), then size floats
            # ============================================================
            uq_size = struct.unpack('Q', f.read(8))[0]  # size_t = 8 bytes (uint64)
            if uq_size > 0:
                f.read(uq_size * 4)  # skip the score floats

            # ============================================================
            # 8. Brain (W_out, gamma, beta)
            # ============================================================
            n = self.vocab_size * self.d_model
            self.W_out = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(self.vocab_size, self.d_model, order='F')
            self.gamma = np.frombuffer(f.read(self.d_model * 4), dtype=np.float32)
            self.beta = np.frombuffer(f.read(self.d_model * 4), dtype=np.float32)

            print(f"[LOADED] vocab={self.vocab_size}, d_model={self.d_model}, d_state={self.d_state}")
            print(f"[LOADED] Embedding: {self.emb_k_shards} shards, {self.emb_m_hash} hash buckets, shard_dim={self.emb_shard_dim}")
            print(f"[LOADED] W_out shape: {self.W_out.shape}, gamma: {self.gamma.shape}, beta: {self.beta.shape}")

    def get_embedding(self, token_id):
        """Replicate C++ CHEEmbedding::get_embedding using the same hash function."""
        result = np.zeros(self.d_model, dtype=np.float32)
        for j in range(self.emb_k_shards):
            # Replicate C++ std::hash<int> with boost-style combine
            h = hash(token_id)
            h ^= hash(j) + 0x9e3779b9 + (h << 6) + (h >> 2)
            h = h % self.emb_m_hash
            start = j * self.emb_shard_dim
            end = start + self.emb_shard_dim
            result[start:end] = self.shards[j][h]
        return result

    def layer_norm(self, x):
        mean = np.mean(x)
        var = np.var(x)
        return self.gamma * ((x - mean) / np.sqrt(var + 1e-6)) + self.beta

    def forward(self, token_id):
        x = self.get_embedding(token_id)

        # SSM
        self.ssm_state = self.ssm_lambda * self.ssm_state + self.ssm_B @ x
        h_ssm = np.real(self.ssm_C @ self.ssm_state) + self.ssm_D * x

        # RFA
        projection = self.rfa_W @ x
        norm_sq = np.sum(x**2)
        phi_x = np.exp(np.clip(projection - norm_sq / 2.0, -20, 20))  # Clip for stability

        self.rfa_num += np.outer(phi_x, x)
        self.rfa_den += phi_x
        h_rfa = (self.rfa_num.T @ phi_x) / (np.dot(self.rfa_den, phi_x) + 1e-6)

        # Core fusion
        h = 0.5 * h_ssm + 0.5 * h_rfa
        h = self.layer_norm(h)

        # Sheaf Reasoning (STRE)
        node = self.stre_W_node @ h
        self.history_nodes.append(node)
        if len(self.history_nodes) > 5:
            self.history_nodes.pop(0)

        # Graph propagation (simplified for inference)
        n = len(self.history_nodes)
        nodes = self.history_nodes
        if n >= 2:
            prev_idx = (n - 2) % n
            next_idx = 0
            h_reason = nodes[-1] + self.stre_W_res @ nodes[prev_idx] + self.stre_W_res @ nodes[next_idx]
            h_reason = np.tanh(h_reason)
        else:
            h_reason = np.tanh(nodes[-1])

        # Synthesis (SSOG) - top-k routing with k=2
        gate_logits = self.ssog_W_gate @ h_reason
        top_k_idx = np.argsort(gate_logits)[-2:]

        h_final = np.zeros(self.d_model)
        for idx in top_k_idx:
            weight = np.exp(gate_logits[idx])
            h_final += weight * (self.experts[idx] @ h_reason)

        # Final Logits
        logits = self.W_out @ h_final
        return logits

    def generate_stream(self, text, max_len=50):
        # Reset state for each generation
        self.ssm_state = np.zeros(self.d_state, dtype=np.complex64)
        self.rfa_num = np.zeros((self.rfa_features, self.d_model), dtype=np.float32)
        self.rfa_den = np.zeros(self.rfa_features, dtype=np.float32)
        self.history_nodes = []

        tokens = text.lower().split()
        ids = [self.vocab.get(t, 1) for t in tokens]

        # Feed context
        for idx in ids[:-1]:
            self.forward(idx)

        curr_id = ids[-1]
        for _ in range(max_len):
            logits = self.forward(curr_id)
            next_id = np.argmax(logits)
            if next_id < 4:
                break
            token = self.id_to_token.get(next_id, "<unk>")
            yield token
            curr_id = next_id


if __name__ == "__main__":
    inf = AlienInference("vocab.txt", "model_checkpoint.bin")
    print(inf.generate("Once upon a time"))
