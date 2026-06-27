#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <string>
#include <iomanip>

class ProgressBar {
public:
    ProgressBar(int total, int width = 50, const std::string& prefix = "Progress") 
        : total(total), width(width), prefix(prefix), current(0) {}

    void update(int value) {
        current = value;
        float progress = (float)current / total;
        int pos = width * progress;

        std::cout << "\r" << prefix << " [";
        for (int i = 0; i < width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "% (" << current << "/" << total << ")" << std::flush;
        
        if (current >= total) std::cout << std::endl;
    }

    void increment() {
        update(current + 1);
    }

private:
    int total;
    int width;
    std::string prefix;
    int current;
};

#endif // UTILS_HPP
