import numpy as np
import os
import struct

class AlienInference:
    def __init__(self, vocab_path, model_path):
        self.load_vocab(vocab_path)
        self.load_model(model_path)
        self.history = []
        self.d_model = 256
        self.d_state = 32

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
            # 0. Global Headers
            self.vocab_size = struct.unpack('i', f.read(4))[0]
            self.d_model = struct.unpack('i', f.read(4))[0]
            self.d_state = 32 # Balanced default for Blitz mode
            
            # 1. Embedding
            self.emb_d_model = struct.unpack('i', f.read(4))[0]
            self.emb_k_shards = struct.unpack('i', f.read(4))[0]
            self.emb_m_hash = struct.unpack('i', f.read(4))[0]
            self.emb_shard_dim = struct.unpack('i', f.read(4))[0]
            
            self.shards = []
            for _ in range(self.emb_k_shards):
                shard_data = np.frombuffer(f.read(self.emb_m_hash * self.emb_shard_dim * 4), dtype=np.float32)
                self.shards.append(shard_data.reshape(self.emb_m_hash, self.emb_shard_dim, order='F'))

            # 2. SSM (S4D)
            lambda_data = np.frombuffer(f.read(self.d_state * 2 * 4), dtype=np.float32)
            self.ssm_lambda = lambda_data[::2] + 1j * lambda_data[1::2]
            
            self.ssm_B = np.frombuffer(f.read(self.d_state * self.d_model * 2 * 4), dtype=np.float32).view(np.complex64).reshape(self.d_state, self.d_model, order='F')
            self.ssm_C = np.frombuffer(f.read(self.d_model * self.d_state * 2 * 4), dtype=np.float32).view(np.complex64).reshape(self.d_model, self.d_state, order='F')
            self.ssm_D = np.frombuffer(f.read(self.d_model * 4), dtype=np.float32)
            self.ssm_state = np.zeros(self.d_state, dtype=np.complex64)

            # 3. RFA
            rfa_features = 16
            self.rfa_W = np.frombuffer(f.read(rfa_features * self.d_model * 4), dtype=np.float32).reshape(rfa_features, self.d_model, order='F')
            self.rfa_num = np.zeros((rfa_features, self.d_model), dtype=np.float32)
            self.rfa_den = np.zeros(rfa_features, dtype=np.float32)

            # 4. STRE (Sheaf)
            self.stre_W_res = np.frombuffer(f.read(self.d_state * self.d_state * 4), dtype=np.float32).reshape(self.d_state, self.d_state, order='F')
            self.stre_W_node = np.frombuffer(f.read(self.d_state * self.d_model * 4), dtype=np.float32).reshape(self.d_state, self.d_model, order='F')

            # 5. ATA
            self.ata_W = np.frombuffer(f.read(self.d_model * self.d_model * 16 * 4), dtype=np.float32).reshape(self.d_model, self.d_model, 16, order='F')

            # 6. SSOG (Synthesizer/Experts)
            n_experts = 8
            self.experts = []
            for _ in range(n_experts):
                self.experts.append(np.frombuffer(f.read(self.d_model * self.d_state * 4), dtype=np.float32).reshape(self.d_model, self.d_state, order='F'))
            self.ssog_W_gate = np.frombuffer(f.read(n_experts * self.d_state * 4), dtype=np.float32).reshape(n_experts, self.d_state, order='F')

            # 7. UQ
            self.uq_W = np.frombuffer(f.read(self.d_model * 1 * 4), dtype=np.float32).reshape(1, self.d_model, order='F')

            # 8. Brain
            self.W_out = np.frombuffer(f.read(self.vocab_size * self.d_model * 4), dtype=np.float32).reshape(self.vocab_size, self.d_model, order='F')
            self.gamma = np.frombuffer(f.read(self.d_model * 4), dtype=np.float32)
            self.beta = np.frombuffer(f.read(self.d_model * 4), dtype=np.float32)


            self.gamma = np.frombuffer(f.read(256 * 4), dtype=np.float32)
            self.beta = np.frombuffer(f.read(256 * 4), dtype=np.float32)

    def hash_func(self, token_id, shard_idx):
        import hashlib
        # Replicate C++ std::hash logic as closely as possible
        h = hash((token_id, shard_idx))
        return h % self.emb_m_hash

    def get_embedding(self, token_id):
        res = np.zeros(self.emb_d_model, dtype=np.float32)
        import zlib
        for j in range(self.emb_k_shards):
            # C++ style hash
            seed = str(token_id).encode() + str(j).encode()
            h = zlib.adler32(seed) % self.emb_m_hash
            res[j*self.emb_shard_dim : (j+1)*self.emb_shard_dim] = self.shards[j][h]
        return res

    def layer_norm(self, x):
        mean = np.mean(x)
        var = np.var(x)
        return self.gamma * ((x - mean) / np.sqrt(var + 1e-6)) + self.beta

    def forward(self, token_id):
        x = self.get_embedding(token_id)
        
        # SSM
        self.ssm_state = self.ssm_lambda * self.ssm_state + self.ssm_B @ x
        h_ssm = np.real(self.ssm_C @ self.ssm_state) + self.ssm_D * x
        
        # RFA (Approximate for inference)
        # phi(x) = exp(Wx - |x|^2/2)
        phi_x = np.exp(self.rfa_W @ x - np.sum(x**2)/2.0)
        self.rfa_num += np.outer(phi_x, x)
        self.rfa_den += phi_x
        h_rfa = (self.rfa_num.T @ phi_x) / (np.dot(self.rfa_den, phi_x) + 1e-6)

        # Core fusion
        h = 0.5 * h_ssm + 0.5 * h_rfa
        h = self.layer_norm(h)

        # Sheaf Reasoning (STRE)
        self.history.append(h)
        if len(self.history) > 5: self.history.pop(0)
        
        # Simple graph propagation
        n = len(self.history)
        nodes = [self.stre_W_node @ node for node in self.history]
        next_nodes = []
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            val = nodes[i] + self.stre_W_res @ nodes[prev_idx] + self.stre_W_res @ nodes[next_idx]
            next_nodes.append(np.tanh(val))
        
        h_reason = next_nodes[-1]

        # Synthesis (SSOG)
        gate = self.ssog_W_gate @ h_reason
        top_k = np.argsort(gate)[-2:]
        h_final = np.zeros(256)
        for idx in top_k:
            weight = np.exp(gate[idx])
            h_final += weight * (self.experts[idx] @ h_reason)
            
        # Logits
        logits = self.W_out @ h_final
        return logits

    def generate(self, text, max_len=20):
        tokens = text.lower().split()
        ids = [self.vocab.get(t, 1) for t in tokens]
        
        for idx in ids[:-1]:
            self.forward(idx)
            
        curr_id = ids[-1]
        output = []
        for _ in range(max_len):
            logits = self.forward(curr_id)
            # Greedy
            next_id = np.argmax(logits)
            if next_id < 4: break
            output.append(self.id_to_token[next_id])
            curr_id = next_id
            
        return " ".join(output)

if __name__ == "__main__":
    inf = AlienInference("vocab.txt", "model_checkpoint.bin")
    print(inf.generate("Once upon a time"))
