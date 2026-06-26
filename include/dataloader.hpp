#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DataLoader {
public:
    static std::string load_text(const std::string& path, size_t max_chars = 1000000) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return "";
        }
        std::string content;
        char buffer[4096];
        while (file.read(buffer, sizeof(buffer)) && content.size() < max_chars) {
            content.append(buffer, file.gcount());
        }
        if (content.size() < max_chars) {
            content.append(buffer, file.gcount());
        }
        return content;
    }

    static std::vector<std::pair<std::string, std::string>> load_jsonl(const std::string& path, size_t max_items = 1000) {
        std::vector<std::pair<std::string, std::string>> data;
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return data;
        }
        std::string line;
        while (std::getline(file, line) && data.size() < max_items) {
            try {
                auto j = json::parse(line);
                data.push_back({j["prompt"], j["completion"]});
            } catch (...) {
                continue;
            }
        }
        return data;
    }
};

#endif // DATALOADER_HPP
