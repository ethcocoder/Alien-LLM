#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

class Tokenizer {
public:
    Tokenizer() {
        // Basic initialization with some special tokens
        vocab[pad_token] = 0;
        vocab[unk_token] = 1;
        vocab[bos_token] = 2;
        vocab[eos_token] = 3;
        
        id_to_token[0] = pad_token;
        id_to_token[1] = unk_token;
        id_to_token[2] = bos_token;
        id_to_token[3] = eos_token;
        
        next_id = 4;
    }

    void train(const std::string& text) {
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) {
            if (vocab.find(word) == vocab.end()) {
                vocab[word] = next_id;
                id_to_token[next_id] = word;
                next_id++;
            }
        }
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) {
            if (vocab.find(word) != vocab.end()) {
                ids.push_back(vocab[word]);
            } else {
                ids.push_back(vocab[unk_token]);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string text;
        for (int id : ids) {
            if (id_to_token.find(id) != id_to_token.end()) {
                text += id_to_token[id] + " ";
            } else {
                text += unk_token + " ";
            }
        }
        return text;
    }

    size_t vocab_size() const { return vocab.size(); }

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    int next_id;
    
    const std::string pad_token = "<PAD>";
    const std::string unk_token = "<UNK>";
    const std::string bos_token = "<BOS>";
    const std::string eos_token = "<EOS>";
};

#endif // TOKENIZER_HPP
