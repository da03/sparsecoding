#include "SCEngine.hpp"

#include <string>
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <mutex>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <petuum_ps_common/include/petuum_ps.hpp>
#include <util/Eigen/Dense>
#include "tools/context.hpp"

namespace sparsecoding {

    SCEngine::SCEngine() {
        // timer
        initT_ = boost::posix_time::microsec_clock::local_time();
        thread_counter_ = 0;
        lda::Context & context = lda::Context::get_instance();
        client_id_ = context.get_int32("client_id");
        data_file_ = context.get_string("data_file");
        output_path_ = context.get_string("output_path");
        num_clients_ = context.get_int32("num_clients");
        num_worker_threads_ = context.get_int32("num_worker_threads");
        num_epochs_ = context.get_int32("num_epochs");
        minibatch_size_ = context.get_int32("minibatch_size");
        num_eval_minibatch_ = context.get_int32("num_eval_minibatch");
        dictionary_size_ = context.get_int32("dictionary_size");
        init_step_size_B_ = context.get_double("init_step_size_B");
        step_size_offset_B_ = context.get_double("step_size_offset_B");
        step_size_pow_B_ = context.get_double("step_size_pow_B");

        num_iter_S_per_minibatch_ = 
            context.get_int32("num_iter_S_per_minibatch");
        init_step_size_S_ = context.get_double("init_step_size_S");
        step_size_offset_S_ = context.get_double("step_size_offset_S");
        step_size_pow_S_ = context.get_double("step_size_pow_S");
        C_ = context.get_double("c");
        lambda_ = context.get_double("lambda");

        X_matrix_loader_.Load(data_file_, client_id_, num_clients_);
        if (dictionary_size_ == 0)
            dictionary_size_ = X_matrix_loader_.GetN();

        S_matrix_loader_.Load(dictionary_size_, X_matrix_loader_.GetN(), 
                client_id_, num_clients_, -0.0, 0.0);

	    int max_client_n = ceil(float(X_matrix_loader_.GetN()) / num_clients_);
	    int iter_minibatch = 
            max_client_n / num_worker_threads_ / minibatch_size_ + 1;
	    num_eval_per_client_ = 
            (num_epochs_ * iter_minibatch - 1) 
              / num_eval_minibatch_ + 1;
    }

    inline void RegVec(std::vector<float> & inc, int len, float C, 
            std::vector<float> & vec_result) {
        float sum = 0.0;
        for (int i = 0; i < len; i++) {
            sum += inc[i] * inc[i];
        }
        float ratio = (sum > C? sqrt(C / sum): 1.0);
        for (int i = 0; i < len; i++) {
            vec_result[i] = inc[i] * ratio;
        }
    }

    // Stochastic Gradient Descent Optimization
    void SCEngine::Start() {
        // thread id on a client
        int thread_id = thread_counter_++;
        petuum::PSTableGroup::RegisterThread();
        LOG(INFO) << "client " << client_id_ << ", thread " 
            << thread_id << "registers!";

        // Get dictionary table and loss table
        petuum::Table<float> B_table = 
            petuum::PSTableGroup::GetTableOrDie<float>(0);
        petuum::Table<float> loss_table = 
            petuum::PSTableGroup::GetTableOrDie<float>(1);

        // size of matrices
        int m = X_matrix_loader_.GetM();
        int client_n = S_matrix_loader_.GetClientN();

        // petuum table accessor
        petuum::RowAccessor row_acc;
        // column id
        int col_id, col_id_client;


        // Cache dictionary table 
        Eigen::MatrixXf petuum_table_cache(m, dictionary_size_);
        // Accumulate update of dictionary table in minibatch
        Eigen::MatrixXf petuum_update_cache(m, dictionary_size_);
        // Cache a column of coefficients S
	    Eigen::VectorXf Sj(dictionary_size_);
        // Cache a column of update of S_j
	    Eigen::VectorXf Sj_inc(dictionary_size_);
        // Cache a column of data X 
	    Eigen::VectorXf Xj(m);
	    Eigen::VectorXf Xj_inc(m);
        // Cache a row of dictionary table
        std::vector<float> petuum_row_cache(m);
	
        // initialize B
        STATS_APP_INIT_BEGIN();
        if (client_id_ == 0 && thread_id == 0) {
            LOG(INFO) << "starting to initialize B";
        }
        if (client_id_ == 0 && thread_id == 0) {
            if (false) {
	        // Load B
	        std::string str, str2;
	        std::ifstream fout_B;
            std::vector<float> B_row_cache(m);

            str = output_path_;
	        str2 = "/B_cache.txt";
                str = str + str2;
                fout_B.open(str.c_str());
                for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                    B_table.Get(row_id, &row_acc);
                    petuum::UpdateBatch<float> B_update;
                    for (int col_id = 0; col_id < m; ++col_id) {
                        fout_B >> B_row_cache[col_id];
			            B_update.Update(col_id, B_row_cache[col_id]);
                    }
		            B_table.BatchInc(row_id, B_update);
                }
                fout_B.close();
		    // Load S
       		if (thread_id == 0) {
		        std::string str, str2;
		        std::ifstream fout_S;
                std::vector<float> S_cache(dictionary_size_), 
                    S_inc_cache(dictionary_size_);

       		    str = output_path_;
       		    char strtemp[10];
       		    sprintf(strtemp, "/S_cache%d.txt", client_id_);
       		    str2 = strtemp;
       		    str = str + str2;
       		    fout_S.open(str.c_str());
       		    for (int col_id_client = 0; col_id_client < client_n; 
                        ++col_id_client) {
       		        if (S_matrix_loader_.GetCol(col_id_client, 
                                col_id, S_cache)) {
       		            for (int row_id = 0; row_id < dictionary_size_; 
                                ++row_id) {
       		                fout_S >> S_inc_cache[row_id];
				            S_inc_cache[row_id] = 
                                S_inc_cache[row_id] - S_cache[row_id];
       		            }
			            S_matrix_loader_.IncCol(col_id_client, S_inc_cache);
       		        }
       		    }
       		    fout_S.close();
       		}
	    } else { // temporarily inactivate 
            srand((unsigned)time(NULL));
            std::vector<float> B_row_cache(m);
            for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                petuum::UpdateBatch<float> B_update;
                B_table.Get(row_id, &row_acc);
                double sum = 0.0;
                for (int col_id = 0; col_id < m; ++col_id) {
                    B_row_cache[col_id] = double(rand()) / RAND_MAX * 2.0 - 1.0;
                    sum += pow(B_row_cache[col_id], 2);
                }
                for (int col_id = 0; col_id < m; ++col_id) {
                    B_row_cache[col_id] *= sqrt(C_ / sum);
                }
                for (int col_id = 0; col_id < m; ++col_id) {
                    B_update.Update(col_id, B_row_cache[col_id]);
                }
                B_table.BatchInc(row_id, B_update);
            }
	    }
        }
        if (thread_id == 0 && client_id_ == 0) {
            LOG(INFO) << "matrix B initialization finished!";
        }
        petuum::PSTableGroup::GlobalBarrier();
        STATS_APP_INIT_END();

        // Optimization Loop
        // Timer
        boost::posix_time::ptime beginT = 
            boost::posix_time::microsec_clock::local_time();
        // Step size for optimization
        float step_size_B = init_step_size_B_, step_size_S = init_step_size_S_;

        for (int iter = 0; iter < num_epochs_; ++iter) {
	        if (thread_id == 0)
                LOG(INFO) << "client: " << client_id_ << "iteration: "<< iter;

            // how many minibatches per epoch
            int minibatch_per_epoch = (client_n / num_worker_threads_ > 0)? 
                client_n / num_worker_threads_: 1;
            for (int num_minibatch = 0; num_minibatch * minibatch_size_ 
                    < minibatch_per_epoch; ++num_minibatch) {
	    	    boost::posix_time::time_duration runTime = 
                    boost::posix_time::microsec_clock::local_time() - initT_;
                // Terminate and save states to disk
		        if ((float) runTime.total_milliseconds() > 13*3600*1000) {
		            LOG(INFO) << "Maximum runtime limit activates, "
                        "terminating now!";
                    petuum::PSTableGroup::GlobalBarrier();
		            if (client_id_ == 0 && thread_id == 0) {
                        std::vector<float> B_row_cache(m);
		                std::string str, str2;
		                std::ofstream fout_B;
                        str = output_path_;
	                    str2 = "/B_cache.txt";
                        str = str + str2;
                        fout_B.open(str.c_str());
                        for (int row_id = 0; row_id < dictionary_size_; 
                                ++row_id) {
                            B_table.Get(row_id, &row_acc);
                            const petuum::DenseRow<float> & petuum_row = 
                                row_acc.Get<petuum::DenseRow<float> >();
                            petuum_row.CopyToVector(&petuum_row_cache);
                            // Regularize by C_
                            RegVec(petuum_row_cache, m, C_, B_row_cache);
                            for (int col_id = 0; col_id < m; ++col_id) {
                                fout_B << B_row_cache[col_id] << "\t";
                            }
                            fout_B << "\n";
                        }
            		    fout_B.close();
		            }
       		        if (thread_id == 0) {
			            std::string str, str2;
			            std::ofstream fout_S;
       		            char strtemp[10];
                        std::vector<float> S_cache(dictionary_size_);

       		            str = output_path_;
       		            sprintf(strtemp, "/S_cache%d.txt", client_id_);
       		            str2 = strtemp;
       		            str = str + str2;
       		            fout_S.open(str.c_str());
       		            for (int col_id_client = 0; col_id_client < client_n; 
                                ++col_id_client) {
       		                if (S_matrix_loader_.GetCol(col_id_client, 
                                    col_id, S_cache)) {
       		                    for (int row_id = 0; row_id < dictionary_size_; 
                                    ++row_id) {
       		                    fout_S << S_cache[row_id] << "\t";
       		                    }
       		                    fout_S << "\n";
       		                }
       		            }
       		            fout_S.close();
       		        }
                    petuum::PSTableGroup::DeregisterThread();
		            return;
		        }
                // Update petuum table cache
		        LOG(INFO) << "starting update table cache";
                for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                    B_table.Get(row_id, &row_acc);
                    const petuum::DenseRow<float> & petuum_row = 
                        row_acc.Get<petuum::DenseRow<float> >();
                    petuum_row.CopyToVector(&petuum_row_cache);
                    for (int col_id = 0; col_id < m; ++col_id) {
                        petuum_table_cache(col_id, row_id) = 
                            petuum_row_cache[col_id];
                    }
                }
		        LOG(INFO) << "finished starting update table cache";
		        // evaluate obj
		        if (num_minibatch % num_eval_minibatch_ == 0) {
	    	        boost::posix_time::time_duration elapTime = 
                        boost::posix_time::microsec_clock::local_time() - beginT;
                    LOG(INFO) <<"evaluating obj";
            	    // evaluate partial obj
                    double obj = 0.0;
                    double obj1 = 0.0;
                    double obj2 = 0.0;
		            int num_samples = (client_n > 250)? 250: client_n;
                    for (int i = 0; i < num_samples; i++) {
                        if (S_matrix_loader_.GetRandCol(col_id_client, 
                                    col_id, Sj) && X_matrix_loader_.GetCol(
                                        col_id_client, col_id, Xj)) {
                            // X_j - B * S_j
				            Xj_inc = Xj - petuum_table_cache * Sj;
				            obj1 += Xj_inc.squaredNorm();
				            obj2 += lambda_ * Sj.lpNorm<1>();
                        }
                    }
		            obj = (obj1+obj2) / num_samples;
                    LOG(ERROR) << "minibatch: " << num_minibatch << ", client " 
                        << client_id_ << "/" << num_clients_ << 
                        " average objective: " << obj;
                    LOG(ERROR) << "minibatch: " << num_minibatch << ", client " 
                        << client_id_ << "/" << num_clients_ <<" obj 1: " << 
                        obj1 / num_samples << " obj2: "<< obj2/num_samples;
                    // update loss table
                    loss_table.Inc(client_id_ * num_eval_per_client_ + 
                            num_minibatch / num_eval_minibatch_,  0, 
                            obj/num_worker_threads_);
		            loss_table.Inc((num_clients_+client_id_) * num_eval_per_client_ 
                            + num_minibatch / num_eval_minibatch_, 0, 
                            ((float) elapTime.total_milliseconds()) / 1000 
                            / num_worker_threads_);
        	    	beginT = boost::posix_time::microsec_clock::local_time();
                        LOG(INFO) <<"finished evaluating obj";
		        }
                step_size_B = init_step_size_B_ * 
                    pow(step_size_offset_B_ + num_minibatch, 
                            -1*step_size_pow_B_);
                step_size_S = init_step_size_S_ * 
                    pow(step_size_offset_S_ + num_minibatch, 
                            -1*step_size_pow_S_);
		        num_minibatch++;
            	// clear update table
            	for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                    for (int col_id = 0; col_id < m; ++col_id) {
                        petuum_update_cache(col_id, row_id) = 0.0;
                    }
            	}
                // mini batch
		        LOG(INFO)<<"starting minibatch" << num_minibatch << ", client "
                    << client_id_ << ", thread " << thread_id;
                for (int k = 0; k < minibatch_size_; ++k) {
                    if (S_matrix_loader_.GetRandCol(col_id_client, col_id, Sj)
                            && X_matrix_loader_.GetCol(col_id_client, col_id, 
                                Xj)) {
                        // update S_j
                        for (int iter_S = 0; 
                                iter_S < num_iter_S_per_minibatch_; ++iter_S) {
		                    Sj_inc = step_size_S * (petuum_table_cache.transpose() 
                                    * (Xj - petuum_table_cache * Sj) - lambda_ 
                                    * (Sj.array().abs() > INFINITESIMAL)
                                    .matrix().cast<float>() );
                            S_matrix_loader_.IncCol(col_id_client, Sj_inc);
			                // get updated S_j
			                S_matrix_loader_.GetCol(col_id_client, col_id, Sj);
                        }
                        // X_j - B * S_j
			            Xj_inc = Xj - petuum_table_cache * Sj;
			            petuum_update_cache.noalias() += 
                            step_size_B * Xj_inc * Sj.transpose();
                    }
                }
		        // calculate updates
		        LOG(INFO)<<"start update B table";
                // Update B_table
                for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                    petuum::UpdateBatch<float> B_update;
                    for (int col_id = 0; col_id < m; ++col_id) {
                        B_update.Update(col_id, 
                                petuum_update_cache(col_id, row_id) 
                                / minibatch_size_);
                    }
                    B_table.BatchInc(row_id, B_update);
                }
                petuum::PSTableGroup::Clock();
                // Update B_table to normalize
                std::vector<float> B_row_cache(m);
                for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                    B_table.Get(row_id, &row_acc);
                    const petuum::DenseRow<float> & petuum_row = 
                        row_acc.Get<petuum::DenseRow<float> >();
                    petuum_row.CopyToVector(&petuum_row_cache);
                    RegVec(petuum_row_cache, m, C_, B_row_cache);
                    petuum::UpdateBatch<float> B_update;
                    for (int col_id = 0; col_id < m; ++col_id) {
                        B_update.Update(col_id, 
                                (-1.0 * petuum_row_cache[col_id] + 
                                B_row_cache[col_id]) / num_clients_ / 
                                num_worker_threads_);
                    }
                    B_table.BatchInc(row_id, B_update);
                }
		        LOG(INFO)<<"update B table ends";
                petuum::PSTableGroup::Clock(); 
            }
        }
        // output result
        std::vector<float> B_row_cache(m), S_cache(dictionary_size_);
        petuum::PSTableGroup::GlobalBarrier();
        std::ofstream fout_loss, fout_B, fout_S, fout_time;
        std::string str, str2;
        if (client_id_ == 0 && thread_id == 0) {
            str = output_path_;
	        str2 = "/loss.txt";
            str = str + str2;
            fout_loss.open(str.c_str());
            LOG(INFO) << "Starting output result: " << std::endl;
            fout_loss << "Loss function evaluated on different clients:" << "\n";
            for (int client = 0; client < num_clients_; client++) {
                fout_loss << "client " << client << "\t";
            }
            fout_loss << "\n";
            for (int iter = 0; iter < num_eval_per_client_; ++iter) {
                for (int client = 0; client < num_clients_; ++client) {
                    int row_id = client * num_eval_per_client_ + iter;
                    loss_table.Get(row_id, &row_acc);
                    const petuum::DenseRow<float> & petuum_row = 
                        row_acc.Get<petuum::DenseRow<float> >();
                    petuum_row.CopyToVector(&petuum_row_cache);
                    fout_loss << petuum_row_cache[0] << "\t";
                }
                fout_loss << "\n";
            }
            fout_loss.close();
            str = output_path_;
	        str2 = "/time.txt";
            str = str + str2;
            fout_time.open(str.c_str());
            for (int iter = 0; iter < num_eval_per_client_; ++iter) {
                for (int client = 0; client < num_clients_; ++client) {
                    int row_id = 
                        (client + num_clients_) * num_eval_per_client_ + iter;
                    loss_table.Get(row_id, &row_acc);
                    const petuum::DenseRow<float> & petuum_row = 
                        row_acc.Get<petuum::DenseRow<float> >();
                    petuum_row.CopyToVector(&petuum_row_cache);
                    fout_time << petuum_row_cache[0] << "\t";
                }
                fout_time << "\n";
            }
	        fout_time.close();
            str = output_path_;
	        str2 = "/B.txt";
            str = str + str2;
            fout_B.open(str.c_str());
            fout_B << "B:\n";
            for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                B_table.Get(row_id, &row_acc);
                const petuum::DenseRow<float> & petuum_row = 
                    row_acc.Get<petuum::DenseRow<float> >();
                petuum_row.CopyToVector(&petuum_row_cache);
                // Regularize by C_
                RegVec(petuum_row_cache, m, C_, B_row_cache);
                for (int col_id = 0; col_id < m; ++col_id) {
                    fout_B << B_row_cache[col_id] << "\t";
                }
                fout_B << "\n";
            }
            fout_B.close();
        }
        if (thread_id == 0) {
            str = output_path_;
            char strtemp[10];
            sprintf(strtemp, "/S%d.txt", client_id_);
            str2 = strtemp;
            str = str + str2;
            fout_S.open(str.c_str());
            for (int col_id_client = 0; col_id_client < client_n; 
                    ++col_id_client) {
                if (S_matrix_loader_.GetCol(col_id_client, col_id, S_cache)) {
                    for (int row_id = 0; row_id < dictionary_size_; ++row_id) {
                        fout_S << S_cache[row_id] << "\t";
                    }
                    fout_S << "\n";
                }
            }
            fout_S.close();
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
} // namespace sparsecoding
