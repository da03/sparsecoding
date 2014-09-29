#include "SCEngine.hpp"
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "tools/context.hpp"
#include <string>

namespace sparsecoding {
    SCEngine::SCEngine() {
        lda::Context & context = lda::Context::get_instance();
        client_id = context.get_int32("client_id");
        std::string data_file;
        data_file = context.get_string("data_file");
        matrix_loader_.Load(data_file, client_id);
    }
    void SCEngine::Start() {
        petuum::PSTableGroup::RegisterThread();
        petuum::PSTableGroup::DeregisterThread();
    }
    int SCEngine::getM() {
        return 0;
    }
    int SCEngine::getN() {
        return 0;
    }
    SCEngine::~SCEngine() {
    }
}
