#pragma once
#include<string>
#include<atomic>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "matrix_loader.hpp"

namespace sparsecoding {
class SCEngine {
    public:
        SCEngine();
        ~SCEngine();
        void Start();
    private:
        std::atomic<int> thread_counter_;

        // petuum parameters
        int client_id_, num_clients_, num_worker_threads_;
        
        // objective function parameters
        int dictionary_size_;
        float C_, lambda_;

        // minibatch and evaluate parameters
        int num_epochs_, minibatch_size_, num_eval_minibatch_, 
            num_iter_S_per_minibatch_,
            num_eval_samples_, num_eval_per_client_;

        // optimization parameters
        float init_step_size_B_, step_size_offset_B_, step_size_pow_B_, 
              init_step_size_S_, step_size_offset_S_, step_size_pow_S_;

        // input and output
        std::string data_file_, data_format_, output_path_;
        bool is_partitioned_;

        // matrix loader for data X and dictionary S
        MatrixLoader<float> X_matrix_loader_, S_matrix_loader_;

        // timer
        boost::posix_time::ptime initT_;
};
}; // namespace sparsecoding
