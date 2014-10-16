#include "SCEngine.hpp"
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "tools/context.hpp"
#include <string>
#include <glog/logging.h>
#include <algorithm>
#include <fstream>

namespace sparsecoding {

    SCEngine::SCEngine() {
        thread_counter_ = 0;
        lda::Context & context = lda::Context::get_instance();
        client_id_ = context.get_int32("client_id");
        data_file_ = context.get_string("data_file");
        num_clients_ = context.get_int32("num_clients");
        num_worker_threads_ = context.get_int32("num_worker_threads");
        num_iterations_per_thread_ = context.get_int32("num_iterations_per_thread");
        mini_batch_ = context.get_int32("mini_batch");
        dictionary_size_ = context.get_int32("dictionary_size");
        init_step_size_ = context.get_double("init_step_size");
        step_size_offset_ = context.get_double("step_size_offset");
        step_size_pow_ = context.get_double("step_size_pow");
        C_ = context.get_double("c");
        lambda_ = context.get_double("lambda");
        X_matrix_loader_.Load(data_file_, client_id_, num_clients_);
        if (dictionary_size_ == 0)
            dictionary_size_ = X_matrix_loader_.GetN();
        S_matrix_loader_.Load(dictionary_size_, X_matrix_loader_.GetN(), client_id_, num_clients_, 0.0, 1.0);
    }

    inline void RegVec(std::vector<float> & inc, int len, float C, float & reg, std::vector<float> & vec_result) {
        float sum = 0.0;
        for (int i = 0; i < len; i++) {
            sum += inc[i] * inc[i];
        }
        //float ratio = (sum > C? sqrt(C / sum): 1.0);
        float ratio = sqrt(C / sum);
        if (sum < INFINITESIMAL)
            ratio = 1.0;
        reg = 1.0 / ratio;
        for (int i = 0; i < len; i++) {
            vec_result[i] = inc[i] * ratio;
        }
    }

    inline void IncVec(std::vector<float> & vec, std::vector<float> & inc, float s) {
        for (std::vector<float>::iterator it = vec.begin(), it2 = inc.begin(); it != vec.end(); it++, it2++) {
            *it += ((*it2) * s);
        }
    }

    inline void VecMinus(std::vector<float> & vec1, std::vector<float> & vec2, std::vector<float> & vec_result) {
        int l = vec1.size();
        for (int i = 0; i < l; i++) {
            vec_result[i] = vec1[i] - vec2[i];
        }
    }

    void SCEngine::Start() {
        int thread_id = thread_counter_++;
        petuum::PSTableGroup::RegisterThread();
        LOG(INFO) << "client " << client_id_ << ", thread " << thread_id << "registers!";

        int m = X_matrix_loader_.GetM();
        int client_n = S_matrix_loader_.GetClientN();
        std::vector<float> X_cache(m), S_cache(dictionary_size_), reg_cache(dictionary_size_);
        std::vector<float> S_inc_cache(dictionary_size_), petuum_row_cache(m), temp_cache(m), temp2_cache(m);
        int col_ind, col_ind_client;
        float **petuum_table_cache, **petuum_update_cache;

        petuum::Table<float> B_table = petuum::PSTableGroup::GetTableOrDie<float>(0);
        petuum::Table<float> loss_table = petuum::PSTableGroup::GetTableOrDie<float>(1);
        //petuum::Table<float> S_table = petuum::PSTableGroup::GetTableOrDie<float>(2);
        petuum::RowAccessor row_acc;

        // allocate space for cache table
        petuum_table_cache = new float *[dictionary_size_];
        petuum_update_cache = new float *[dictionary_size_];
        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
            petuum_table_cache[row_ind] = new float[m];
            petuum_update_cache[row_ind] = new float[m];
        }

        // initialize petuum table cache
        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
            for (int col_ind = 0; col_ind < m; col_ind++) {
                petuum_table_cache[row_ind][col_ind] = 0.0;
            }
        }
        // initialize B
        STATS_APP_INIT_BEGIN();
        if (client_id_ == 0 && thread_id == 0) {
            LOG(INFO) << "starting to initialize B";
        }
        if (client_id_ == 0 && thread_id == 0) {
            srand((unsigned)time(NULL));
            for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                B_table.Get(row_ind, &row_acc);
                petuum::UpdateBatch<float> B_update;
                double sum = 0.0;
                for (int col_ind = 0; col_ind < m; col_ind++) {
                    temp2_cache[col_ind] = double(rand()) / RAND_MAX * 2.0 - 1.0;
                    sum += pow(temp2_cache[col_ind], 2);
                }
                for (int col_ind = 0; col_ind < m; col_ind++) {
                    temp2_cache[col_ind] *= sqrt(C_ / sum);
                }
                for (int col_ind = 0; col_ind < m; col_ind++) {
                    B_update.Update(col_ind, temp2_cache[col_ind]);
                }
                B_table.BatchInc(row_ind, B_update);
            }
        }
        if (thread_id == 0 && client_id_ == 0) {
            LOG(INFO) << "matrix B initialization finished!";
        }
        petuum::PSTableGroup::GlobalBarrier();
        STATS_APP_INIT_END();

        int t = 0;
        float step_size = init_step_size_;
        // start iterations
        for (int iter = 0; iter < num_iterations_per_thread_; iter++) {
            LOG(INFO) << "client: " << client_id_ << "iteration: "<< iter;
            // cache petuum table
            for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                    B_table.Get(row_ind, &row_acc);
                    const petuum::DenseRow<float> & petuum_row = row_acc.Get<petuum::DenseRow<float> >();
                    petuum_row.CopyToVector(&petuum_row_cache);
                    for (int col_ind = 0; col_ind < m; col_ind++) {
                        petuum_table_cache[row_ind][col_ind] = petuum_row_cache[col_ind];
                    }
            }
            // clear update table
            for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                for (int col_ind = 0; col_ind < m; col_ind++) {
                    petuum_update_cache[row_ind][col_ind] = 0.0;
                }
            }
            int iter_per_thread_B = (client_n / num_worker_threads_ > 0)? client_n / num_worker_threads_: 1;
            // update B given S
            for (int iter_b = 0; iter_b == 0 || iter_b * mini_batch_ < iter_per_thread_B; iter_b++) {
                step_size = init_step_size_ * pow(step_size_offset_ + t, -1*step_size_pow_);
                t++;
                // mini batch
                for (int k = 0; k < mini_batch_; k++) {
                    if (S_matrix_loader_.GetRandCol(col_ind_client, col_ind, S_cache) && X_matrix_loader_.GetCol(col_ind_client, col_ind, X_cache)) {
                        std::fill(temp_cache.begin(), temp_cache.end(), 0);
                        // B * S_j
                        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                            for (int col_ind = 0; col_ind < m; col_ind++) {
                                petuum_row_cache[col_ind] = petuum_table_cache[row_ind][col_ind];
                            }
                            // Regularize by C_
                            RegVec(petuum_row_cache, m, C_, reg_cache[row_ind], temp2_cache);
                            IncVec(temp_cache, temp2_cache, S_cache[row_ind]);
                        }
                        // X_j - B * S_j
                        VecMinus(X_cache, temp_cache, temp2_cache);

                        // Update cached B_table
                        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                            for (int col_ind = 0; col_ind < m; col_ind++) {
                                petuum_row_cache[col_ind] = petuum_table_cache[row_ind][col_ind];
                            }
                            for (int col_ind = 0; col_ind < m; col_ind++) {
                                // regualize or not?
                                temp_cache[col_ind] = petuum_row_cache[col_ind] / reg_cache[row_ind] + step_size * temp2_cache[col_ind] * S_cache[row_ind];
                                //temp_cache[col_ind] = petuum_row_cache[col_ind] + step_size * temp2_cache[col_ind] * S_cache[row_ind] * reg_cache[row_ind];
                            }
                            RegVec(temp_cache, m, C_, reg_cache[row_ind], temp2_cache);
                            for (int col_ind = 0; col_ind < m; col_ind++) {
                                petuum_update_cache[row_ind][col_ind] += -1.0*petuum_row_cache[col_ind] + temp2_cache[col_ind];
                                petuum_table_cache[row_ind][col_ind] += -1.0*petuum_row_cache[col_ind] + temp2_cache[col_ind];
                                //petuum_update_cache[row_ind][col_ind] += -1.0*petuum_row_cache[col_ind] + temp_cache[col_ind];
                                //petuum_table_cache[row_ind][col_ind] += -1.0*petuum_row_cache[col_ind] + temp_cache[col_ind];
                            }
                        }
                    }
                }
            }
            // Update B_table
            for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                petuum::UpdateBatch<float> B_update;
                for (int col_ind = 0; col_ind < m; col_ind++) {
                    B_update.Update(col_ind, petuum_update_cache[row_ind][col_ind]);
                }
                B_table.BatchInc(row_ind, B_update);
            }
            petuum::PSTableGroup::Clock();
            // Update B_table to normalize
            for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                B_table.Get(row_ind, &row_acc);
                const petuum::DenseRow<float> & petuum_row = row_acc.Get<petuum::DenseRow<float> >();
                petuum_row.CopyToVector(&petuum_row_cache);
                RegVec(petuum_row_cache, m, C_, reg_cache[row_ind], temp_cache);
                petuum::UpdateBatch<float> B_update;
                for (int col_ind = 0; col_ind < m; col_ind++) {
                    B_update.Update(col_ind, (-1.0*petuum_row_cache[col_ind] + temp_cache[col_ind])/num_clients_/num_worker_threads_);
                }
                B_table.BatchInc(row_ind, B_update);
            }
            petuum::PSTableGroup::Clock();
                
            // update S given B
            int iter_per_thread_s = (client_n / num_worker_threads_ > 0)? client_n / num_worker_threads_: 1;
            for (int iter_s = 0; iter_s == 0 || iter_s * mini_batch_ < iter_per_thread_s; iter_s++) {
                step_size = init_step_size_ * pow(step_size_offset_ + t, -1*step_size_pow_);
                t++;
                for (int k = 0; k < mini_batch_; k++) {
                    if (S_matrix_loader_.GetRandCol(col_ind_client, col_ind, S_cache) && X_matrix_loader_.GetCol(col_ind_client, col_ind, X_cache)) {
                        std::fill(temp_cache.begin(), temp_cache.end(), 0);
                        // B * S_j
                        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                            for (int col_ind = 0; col_ind < m; col_ind++)
                                petuum_row_cache[col_ind] = petuum_table_cache[row_ind][col_ind];
                            // Regularize by C_
                            RegVec(petuum_row_cache, m, C_, reg_cache[row_ind], temp2_cache);
                            IncVec(temp_cache, temp2_cache, S_cache[row_ind]);
                        }
                        // X_j - B * S_j
                        VecMinus(X_cache, temp_cache, temp2_cache);
                        // B^T * (X_j - B * S_j)
                        std::fill(S_inc_cache.begin(), S_inc_cache.end(), 0);
                        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                            for (int col_ind = 0; col_ind < m; col_ind++)
                                petuum_row_cache[col_ind] = petuum_table_cache[row_ind][col_ind];
                            // Regularize by C_
                            RegVec(petuum_row_cache, m, C_, reg_cache[row_ind], temp_cache);
                            for (int temp = 0; temp < m; temp++) {
                               S_inc_cache[row_ind] += temp2_cache[temp] * temp_cache[temp];
                            }
                            S_inc_cache[row_ind] -= lambda_ * ((S_cache[row_ind] > INFINITESIMAL)? 1.0: 0.0);
                            S_inc_cache[row_ind] -= lambda_ * ((S_cache[row_ind] < -INFINITESIMAL)? -1.0: 0.0);
                            S_inc_cache[row_ind] *= step_size;
                        }
                        // update S_j
                        S_matrix_loader_.IncCol(col_ind_client, S_inc_cache);
                        /*// temporary debugging S
                        for (int row_ind = col_ind; row_ind < col_ind+1; row_ind++) {
                            petuum::UpdateBatch<float> S_update;
                            S_table.Get(row_ind, &row_acc);
                            const petuum::DenseRow<float> & petuum_row = row_acc.Get<petuum::DenseRow<float> >();
                            petuum_row.CopyToVector(&petuum_row_cache);
                            S_matrix_loader_.GetCol(col_ind_client, col_ind, S_cache);
                            for (int col_ind = 0; col_ind < dictionary_size_; col_ind++) {
                                S_update.Update(col_ind, -1*petuum_row_cache[col_ind] + S_cache[col_ind]);
                            }
                            S_table.BatchInc(row_ind, S_update);
                        }*/
                    }

                }
            }
            LOG(INFO) <<"evaluating obj";
            // evaluate partial obj
            if (thread_id == 0) {
                double obj = 0.0;
                for (int i = 0; i < client_n; i++) {
                    col_ind_client = i;
                    if (S_matrix_loader_.GetCol(col_ind_client, col_ind, S_cache) && X_matrix_loader_.GetCol(col_ind_client, col_ind, X_cache)) {
                        std::fill(temp_cache.begin(), temp_cache.end(), 0);
                        // B * S_j
                        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                            for (int col_ind = 0; col_ind < m; col_ind++)
                                petuum_row_cache[col_ind] = petuum_table_cache[row_ind][col_ind];

                            // Regularize by C_
                            RegVec(petuum_row_cache, m, C_, reg_cache[row_ind], temp2_cache);
                            IncVec(temp_cache, temp2_cache, S_cache[row_ind]);
                        }
                        // X_j - B * S_j
                        VecMinus(X_cache, temp_cache, temp2_cache);
                        for (int temp = 0; temp < m; temp++) {
                            obj += temp2_cache[temp] * temp2_cache[temp];
                        }
                        for (int temp = 0; temp < dictionary_size_; temp++) {
                            obj += lambda_ * std::abs(S_cache[temp]);
                        }
                    }
                }
                if (client_id_ == 0)
                    LOG(INFO) << "-----------------------iter: " << iter << " 1 out of " << num_clients_ <<" objective: " << obj;
                // update loss table
                loss_table.Inc(client_id_ * num_iterations_per_thread_ + iter,  0, obj);
            }
            //petuum::PSTableGroup::Clock();
        }
        // output result, temporary version
        petuum::PSTableGroup::GlobalBarrier();
        std::ofstream fout_loss, fout_B, fout_S;
        if (client_id_ == 0 && thread_id == 0) {
            fout_loss.open("/tank/projects/biglearning/pxie/back/dnn_exp_jun_rls/petuum/apps/sparsecoding/data/loss.txt");
            std::cout << "Starting output result: " << std::endl;
            fout_loss << "Loss function evaluated on different clients:" << "\n";
            for (int client = 0; client < num_clients_; client++) {
                fout_loss << "client " << client << "\t";
            }
            fout_loss << "\n";
            for (int iter = 0; iter < num_iterations_per_thread_; iter++) {
                for (int client = 0; client < num_clients_; client++) {
                    int row_ind = client * num_iterations_per_thread_ + iter;
                    loss_table.Get(row_ind, &row_acc);
                    const petuum::DenseRow<float> & petuum_row = row_acc.Get<petuum::DenseRow<float> >();
                    petuum_row.CopyToVector(&petuum_row_cache);
                    fout_loss << petuum_row_cache[0] << "\t";
                }
                fout_loss << "\n";
            }

            fout_loss.close();
            fout_B.open("/tank/projects/biglearning/pxie/back/dnn_exp_jun_rls/petuum/apps/sparsecoding/data/B.txt");
            fout_B << "B:\n";
            for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                B_table.Get(row_ind, &row_acc);
                const petuum::DenseRow<float> & petuum_row = row_acc.Get<petuum::DenseRow<float> >();
                petuum_row.CopyToVector(&petuum_row_cache);
                // Regularize by C_
                RegVec(petuum_row_cache, m, C_, reg_cache[row_ind], temp2_cache);
                for (int col_ind = 0; col_ind < m; col_ind++) {
                    fout_B << temp2_cache[col_ind] << "\t";
                }
                fout_B << "\n";
            }
            fout_B.close();
            /*for (int col_ind = 0; col_ind < dictionary_size_; col_ind++) {
                for (int row_ind = 0; row_ind < GetN(); row_ind++) {
                    S_table.Get(row_ind, &row_acc);
                    const petuum::DenseRow<float> & petuum_row = row_acc.Get<petuum::DenseRow<float> >();
                    std::cout << std::setw(8) << std::fixed << std::setprecision(4) << petuum_row[col_ind];
                }
                std::cout << std::endl;
            }*/
        }
        if (thread_id == 0) {
            std::string str, str2;
            str = "/tank/projects/biglearning/pxie/back/dnn_exp_jun_rls/petuum/apps/sparsecoding/data/B";
            char strtemp[10];
            sprintf(strtemp, "%d.txt", client_id_);
            str2 = strtemp;
            str = str + str2;
            fout_S.open(str.c_str());
            for (int i = 0; i < client_n; i++) {
                col_ind_client = i;
                if (S_matrix_loader_.GetCol(col_ind_client, col_ind, S_cache)) {
                    for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
                        fout_S << S_cache[row_ind] << "\t";
                    }
                    fout_S << "\n";
                }
            }
            fout_S.close();
        }
        // release allocated space for cache table
        for (int row_ind = 0; row_ind < dictionary_size_; row_ind++) {
            delete [] petuum_table_cache[row_ind];
            delete [] petuum_update_cache[row_ind];
        }
        delete [] petuum_table_cache;
        delete [] petuum_update_cache;
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
