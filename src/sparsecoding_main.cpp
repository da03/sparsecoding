#include <petuum_ps_common/include/petuum_ps.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "SCEngine.hpp"
#include <vector>
#include <thread>
#include "tools/context.hpp"

// Petuum Parameters
DEFINE_string(hostfile, "", "Path to file containing server ip:port.");
DEFINE_int32(num_clients, 1, "Total number of clients");
DEFINE_int32(num_worker_threads, 4, "Number of app threads in this client");
DEFINE_int32(client_id, 0, "Client ID");
 
// Sparse Coding Parameters
DEFINE_string(data_file, "", "Input matrix.");
DEFINE_int32(dictionary_size, 0, "Size of dictionary. "
        "Default value is number of rows in input matrix.");
DEFINE_double(lambda, 1.0, "L1 regularization strength. "
        "Default value is 1.0.");
DEFINE_double(c, 1.0, "L2 norm constraint on elements of dictionary. "
        "Default value is 1.0.");
DEFINE_int32(num_iterations_per_thread, 100, 
        "Number of iterations per thread. "
        "Default value is 0.5.");
DEFINE_double(init_step_size, 0.5, "SGD step size at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 0.5.");
DEFINE_double(step_size_offset, 100.0, "SGD step size at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 100.0.");
DEFINE_double(step_size_pow, 0.5, "SGD step size at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 0.5.");

// Misc
DEFINE_int32(table_staleness, 0, "Staleness for dictionary table."
        "Default value is 0.");

// No need to change the following
DEFINE_string(stats_path, "", "Statistics output file.");
DEFINE_string(consistency_model, "SSPPush", "SSP or SSPPush or ...");

int main(int argc, char * argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    petuum::TableGroupConfig table_group_config;
    // 1 server thread per client
    table_group_config.num_total_server_threads = FLAGS_num_clients;
    // 1 background thread per client
    table_group_config.num_total_bg_threads = FLAGS_num_clients;
    table_group_config.num_total_clients = FLAGS_num_clients;
    // dictionary table and loss table
    table_group_config.num_tables = 2;
    table_group_config.num_local_server_threads = 1;
    table_group_config.num_local_bg_threads = 1;
    // + 1 for main()
    table_group_config.num_local_app_threads = FLAGS_num_worker_threads + 1;;
    table_group_config.client_id = FLAGS_client_id;
    
    petuum::GetHostInfos(FLAGS_hostfile, &table_group_config.host_map);
    petuum::GetServerIDsFromHostMap(&table_group_config.server_ids, table_group_config.host_map);
    if (FLAGS_consistency_model == "SSP") {
        table_group_config.consistency_model = petuum::SSP;
    } else if (FLAGS_consistency_model == "SSPPush") {
        table_group_config.consistency_model = petuum::SSPPush;
    } else if (FLAGS_consistency_model == "LocalOOC") {
        table_group_config.consistency_model = petuum::LocalOOC;
    } else {
        LOG(FATAL) << "Unknown consistency model: " << FLAGS_consistency_model;
    }
    // stats
    table_group_config.stats_path = FLAGS_stats_path;
    // configure row types
    petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float> >(0);

    // start PS
    petuum::PSTableGroup::Init(table_group_config, false);

    // load data
    STATS_APP_LOAD_DATA_BEGIN();
    sparsecoding::SCEngine sc_engine;
    LOG(INFO) << "Data loaded!";
    STATS_APP_LOAD_DATA_END();

    // create PS table
    // B_table (dictionary_size by number of rows in input matrix)
    petuum::ClientTableConfig table_config;
    table_config.table_info.row_type = 0;
    table_config.table_info.table_staleness = FLAGS_table_staleness;
    table_config.table_info.row_capacity = (FLAGS_dictionary_size == 0? sc_engine.getN(): FLAGS_dictionary_size);
    // all rows put into memory, to be modified
    table_config.process_cache_capacity = sc_engine.getM();
    CHECK(petuum::PSTableGroup::CreateTable(0, table_config)) << "Failed to create dictionary table";
    // loss table. Single column. Each column is loss in one iteration
    table_config.table_info.row_type = 0;
    table_config.table_info.table_staleness = 0;
    table_config.table_info.row_capacity = 3;
    table_config.process_cache_capacity = FLAGS_num_iterations_per_thread * FLAGS_num_clients;
    CHECK(petuum::PSTableGroup::CreateTable(1, table_config)) << "Failed to create loss table";

    petuum::PSTableGroup::CreateTableDone();

    std::vector<std::thread> threads(FLAGS_num_worker_threads);
    for (auto & thr: threads) {
        thr = std::thread(&sparsecoding::SCEngine::Start, std::ref(sc_engine));
    }
    for (auto & thr: threads) {
        thr.join();
    }
    petuum::PSTableGroup::ShutDown();
    LOG(INFO) << "Sparse Coding shut down!";
    return 0;
}
