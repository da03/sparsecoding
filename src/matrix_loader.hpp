#pragma once
#include<string>
#include<atomic>
#include<vector>

namespace sparsecoding {
class MatrixLoader {
    public:
        MatrixLoader();
        ~MatrixLoader();
        void Load(std::string data_file, int client_id);
    private:
        std::atomic<std::vector<float> > data;
};
};
