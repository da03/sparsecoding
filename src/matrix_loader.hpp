#pragma once
#include<string>
#include<atomic>
#include<vector>
#include<mutex>

// Elements whose absolute value is smaller than INFINITESIMAL stored in matrix 
// would be considered 0 after performing function IncCol() on that column
#define INFINITESIMAL 0.0001
// Elements cannot exceed MAXELEVAL to prevent overflow, 
// also takes effect after performing function IncCol() on that column
#define MAXELEVAL 1000.0
// Elements cannot be smaller than MAXELEVAL to prevent overflow, 
// also takes effect after performing function IncCol() on that column
#define MINELEVAL -1000.0

namespace sparsecoding {
class MatrixLoader {
    public:
        MatrixLoader();
        ~MatrixLoader();
        void Load(std::string data_file, int client_id, int num_clients);
        void Load(int m, int n, int client_id, int num_clients, float low, float high);
        int GetM();
        int GetN();
        int GetClientN();
        bool GetCol(int j_client, int & j, std::vector<float> & col);
        bool GetRandCol(int & j_client, int & j, std::vector<float> & col);
        void IncCol(int j_client, std::vector<float> & inc);
    private:
        std::vector<std::vector<float> > data_;
        int m_, n_;
        int client_id_, client_n_, num_clients_;
        std::mutex * mtx_;
};
};
