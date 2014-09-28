#pragma once
#include<string>

namespace sparsecoding {
class SCEngine {
    public:
        SCEngine();
        ~SCEngine();
        void Start();
        int getM();
        int getN();
    private:
        int client_id;
        std::string data_file;
};
};
