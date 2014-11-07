#include <string>
#include <vector>
#include <cstdio>
#include <mutex>
#include "matrix_loader.hpp"
#include <iostream>
#include <glog/logging.h>

namespace sparsecoding {

MatrixLoader::MatrixLoader() {
    srand((unsigned)time(NULL));
}

void MatrixLoader::Load(std::string data_file, int client_id, int num_clients) {
    FILE * fp;
    float temp;
    fp = fopen(data_file.c_str(), "rb");
    m_ = 21504;
    //n_ = 1266734;
    n_ = 126673;
    num_clients_ = num_clients;
    client_id_ = client_id;
    if (client_id >= n_) {
        client_n_ = 0;
    }
    else {
        client_n_ = (n_ - (n_ / num_clients) * num_clients > client_id)? n_ / num_clients + 1: n_ / num_clients;
        data_.resize(client_n_);
        for (int k = 0; k < client_n_; k++) {
            data_[k].resize(m_);
        }
        for (int j = 0; j < n_; ++j) {
            for (int i = 0; i < m_; ++i) {
                fread(&temp, 4, 1, fp);
                if (j % num_clients == client_id) {
                    data_[j / num_clients][i] = temp;
                }
            }
        }
        mtx_ = new std::mutex[client_n_];
        fclose(fp);
    }
}

void MatrixLoader::Load(int m, int n, int client_id, int num_clients, float low, float high) {
    num_clients_ = num_clients;
    m_ = m;
    n_ = n;
    client_id_ = client_id;
    if (client_id >= n_) {
        client_n_ = 0;
    }
    else {
        srand((unsigned)time(NULL));
        client_n_ = (n_ - (n_ / num_clients) * num_clients > client_id)? n_ / num_clients + 1: n_ / num_clients;
        data_.resize(client_n_);
        for (int k = 0; k < client_n_; k++) {
            data_[k].resize(m_);
            for (int i = 0; i < m_; i++) {
                data_[k][i] = low + float(rand()) / RAND_MAX * (high - low);
            }
        }
        mtx_ = new std::mutex[client_n_];
    }
}

MatrixLoader::~MatrixLoader() {
    if (client_n_ > 0)
        delete [] mtx_;
}

int MatrixLoader::GetM() {
    return m_;
}

int MatrixLoader::GetN() {
    return n_;
}

int MatrixLoader::GetClientN() {
    return client_n_;
}

bool MatrixLoader::GetCol(int j_client, int & j, std::vector<float> & col) {
    if (client_n_ == 0)
        return false;
    std::unique_lock<std::mutex> lck (*(mtx_+j_client));
    col = data_[j_client];
    j = j_client * num_clients_ + client_id_;
    return true;
}

bool MatrixLoader::GetRandCol(int & j_client, int & j, std::vector<float> & col) {
    if (client_n_ == 0)
        return false;
    j_client = rand() % client_n_;
    GetCol(j_client, j, col);
    return true;
}

void MatrixLoader::IncCol(int j_client, std::vector<float> & inc) {
    std::unique_lock<std::mutex> lck (*(mtx_+j_client));
    for (int i = 0; i < m_; i++) {
        data_[j_client][i] += inc[i];
        if ((data_[j_client][i] > -INFINITESIMAL) && (data_[j_client][i] < INFINITESIMAL))
            data_[j_client][i] = 0.0;
        if (data_[j_client][i] > MAXELEVAL)
            data_[j_client][i] = MAXELEVAL;
        if (data_[j_client][i] < MINELEVAL)
            data_[j_client][i] = MINELEVAL;
    }
}
}
