#include <string>
#include <vector>
#include <stdio>
#include "matrix_loader.hpp"

namespace sparsecoding {
MatrixLoader::MatrixLoader() {
}
void MatrixLoader::Load(std::string data_file, int client_id) {
    FILE * fp;
    fp = fopen(data_file.c_str(), 'r');
    fscanf("%d%d", &m, &n);
}
MatrixLoader::~MatrixLoader() {
}
int MatrixLoader::GetM() {
    return m;
}
int MatrixLoader::GetClientN() {
    return client_n;
}
std::vector<float> MatrixLoader::GetData(int i){
}

}
