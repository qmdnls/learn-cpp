#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <file> <search term>\n";
        return 1;
    }

    std::string path = argv[1];
    std::string search_term = argv[2];

    std::ifstream input_file = std::ifstream(path);

    if (!input_file) {
        std::cerr << "cannot open file '" << path << "'\n";
        return 1;
    }

    std::string line;
    size_t line_number = 1;
    
    while (std::getline(input_file, line)) {
        size_t start = 0;
        size_t pos = line.find(search_term);
        bool matched = false;
        
        while (pos != std::string::npos) {
            if (!matched) {
                std::cout << line_number << ": ";
                matched = true;
            }

            std::cout << line.substr(start, pos - start);
            std::cout << "\033[31m" << search_term << "\033[0m"; // match in red 

            start = pos + search_term.size();
            pos = line.find(search_term, start);
            
        }

        if (matched) {
            std::cout << line.substr(start) << "\n";
        }
        
        line_number++;
    }
}
