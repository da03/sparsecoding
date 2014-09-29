#pragma once
#include<string>
#include "matrix_loader.hpp"
#include<atomic>

namespace sparsecoding {
class SCEngine {
    public:
        SCEngine();
        ~SCEngine();
        void Start();
        int getM();
        int getN();
    private:
        std::atomic<int> thread_counter_;
        int client_id_;
        MatrixLoader matrix_loader_;
};
};
