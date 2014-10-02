#include "SCEngine.hpp"
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "tools/context.hpp"
#include <string>
#include <glog/logging.h>
#include <algorithm>

namespace sparsecoding {
    SCEngine::SCEngine() {
        thread_counter_ = 0;
        lda::Context & context = lda::Context::get_instance();
        client_id_ = context.get_int32("client_id");
        data_file_ = context.get_string("data_file");
        num_clients_ = context.get_int32("num_clients");
        num_iterations_per_thread_ = context.get_int32("num_iterations_per_thread");
        mini_batch_ = context.get_int32("mini_batch");
        dictionary_size_ = context.get_int32("dictionary_size");
        init_step_size_ = context.get_double("init_step_size");
        step_size_offset_ = context.get_double("step_size_offset");
        step_size_pow_ = context.get_double("step_size_pow");
        C_ = context.get_double("c");
        lambda_ = context.get_double("lambda");
        X_matrix_loader_.Load(data_file_, client_id_, num_clients_);
        S_matrix_loader_.Load(X_matrix_loader_.GetM(), X_matrix_loader_.GetN(), client_id_, num_clients_, 0.0, 1.0);
    }

    inline std::vector<float> RegVec(const petuum::DenseRow<float> & inc, int len, float C, float & reg) {
        std::vector<float> vec_result;
        float sum = 0.0;
        for (int i = 0; i < len; i++) {
            sum += inc[i] * inc[i];
        }
        float ratio = sum > C? sqrt(C / sum): 1.0;
        reg = 1.0 / ratio;
        for (int i = 0; i < len; i++) {
            vec_result.push_back(inc[i] * ratio);
        }
        return vec_result;
    }

    inline void IncVec(std::vector<float> & vec, std::vector<float> & inc) {
        for (std::vector<float>::iterator it = vec.begin(), it2 = inc.begin(); it != vec.end(); it++, it2++) {
            *it += *it2;
        }
    }

    inline std::vector<float> VecMinus(std::vector<float> & vec1, std::vector<float> & vec2) {
        std::vector<float> vec_result;
        int l = vec1.size();
        for (int i = 0; i < l; i++) {
            vec_result.push_back(vec1[i] - vec2[i]);
        }
        return vec_result;
    }

    void SCEngine::Start() {
        int thread_id = thread_counter_++;
        LOG_IF(INFO, thread_id == 0) << "thread starts!";
        petuum::PSTableGroup::RegisterThread();
        int m = X_matrix_loader_.GetM();
        int client_n = S_matrix_loader_.GetClientN();
        std::vector<float> X_col_cache(m), S_col_cache(m), reg_cache(dictionary_size_);
        std::vector<float> temp_col_cache(m), temp2_col_cache(m);
        int col_ind, col_ind_client;

        petuum::Table<float> B_table = petuum::PSTableGroup::GetTableOrDie<float>(0);
        petuum::RowAccessor Bj_acc;
        petuum::UpdateBatch<float> B_update;

        int t = 0;
        float step_size = init_step_size_;
        // start iterations
        for (int iter = 0; iter < num_iterations_per_thread_; iter++) {
            // update B given S
            for (int iter_b = 0; iter_b * mini_batch_ < client_n; iter_b++) {
                step_size = init_step_size_ * pow(step_size_offset_ + t, -1*step_size_pow_);
                t++;
                for (int k = 0; k < mini_batch_; k++) {
                    if (S_matrix_loader_.GetRandCol(col_ind_client, col_ind, S_col_cache) && X_matrix_loader_.GetCol(col_ind_client, col_ind, X_col_cache)) {
                        std::fill(temp_col_cache.begin(), temp_col_cache.end(), 0);
                        // B * S_j
                        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                            B_table.Get(row_ind, &Bj_acc);
                            const petuum::DenseRow<float> & Bj = Bj_acc.Get<petuum::DenseRow<float> >();
                            // Regularize by C_
                            temp2_col_cache = RegVec(Bj, m, C_, reg_cache[row_ind]);
                            IncVec(temp_col_cache, temp2_col_cache);
                        }
                        // X_j - B * S_j
                        temp_col_cache = VecMinus(X_col_cache, temp_col_cache);

                        // Update B_table
                        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                            for (int col_ind = 0; col_ind < m; col_ind++) {
                                B_update.Update(col_ind, step_size * temp_col_cache[col_ind] * S_col_cache[row_ind] * reg_cache[row_ind]);
                            }
                            B_table.BatchInc(row_ind, B_update);
                        }
                    }
                }
                petuum::PSTableGroup::Clock();
            }
            // update S given B
            for (int iter_s = 0; iter_s * mini_batch_ < client_n; iter_s++) {
                step_size = init_step_size_ * pow(step_size_offset_ + t, -1*step_size_pow_);
                t++;
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
