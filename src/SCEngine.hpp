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
        int GetM();
        int GetN();
    private:
        std::atomic<int> thread_counter_;
        int client_id_, num_clients_;
        int num_iterations_per_thread_, num_worker_threads_, mini_batch_, num_eval_per_client_, num_eval_minibatch_;

        float init_step_size_, step_size_offset_, step_size_pow_;
        float C_, lambda_;
        int dictionary_size_;

        std::string data_file_, output_path_;
        MatrixLoader X_matrix_loader_, S_matrix_loader_;

};
};
