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
        int GetM();
        int GetClientN();
        std::vector<float> GetData(int i);
    private:
        std::vector<std::vector<float> > data;
        int m, n;
        int client_n;
};
};
