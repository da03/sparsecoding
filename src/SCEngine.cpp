#include "SCEngine.hpp"
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "tools/context.hpp"
#include <string>

namespace sparsecoding {
    SCEngine::SCEngine() {
        thread_counter_ = 0;
        lda::Context & context = lda::Context::get_instance();
        client_id_ = context.get_int32("client_id");
        data_file_ = context.get_string("data_file");
        num_clients_ = context.get_int32("num_clients");
        num_iterations_per_thread_ = context.get_int32("num_iterations_per_thread");
        mini_batch_ = context.get_int32("mini_batch");
        init_step_size_ = context.get_double("init_step_size");
        step_size_offset_ = context.get_double("step_size_offset");
        step_size_pow_ = context.get_double("step_size_pow");
        X_matrix_loader_.Load(data_file_, client_id_, num_clients_);
        S_matrix_loader_.Load(X_matrix_loader_.GetM(), X_matrix_loader_.GetN(), client_id_, num_clients_, 0.0, 1.0);
    }
    void SCEngine::Start() {
        thread_counter_++;
        petuum::PSTableGroup::RegisterThread();
        int m = X_matrix_loader_.GetM();
        int client_n = S_matrix_loader_.GetClientN();
        std::vector<float> x_col_cache(m);
        float alpha = init_step_size_;
        // start iterations
        for (int iter = 0; iter < num_iterations_per_thread_; iter++) {
            // update B given S
            for (int iter_b = 0; iter_b < client_n; iter_b++) {
                for (int k = 0; k < mini_batch_; k++) {
                }
            }
            // update S given B
            for (int iter_s = 0; iter_s < client_n; iter_s++) {
                for (int k = 0; k < mini_batch_; k++) {
                }
            }
        }
        petuum::PSTableGroup::DeregisterThread();
    }
    int SCEngine::GetM() {
        return X_matrix_loader_.GetM();
    }
    int SCEngine::GetN() {
        return X_matrix_loader_.GetN();
    }
    SCEngine::~SCEngine() {
    }
}
