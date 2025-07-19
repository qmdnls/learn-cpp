#include <iostream>
#include <vector>
#include <cmath>

using Vec = std::vector<float>; // single float vector
using Mat = std::vector<Vec>; // row-major Matrix

Vec matvec_mul(const Mat& W, const Vec& x) {
    Vec result(W.size(), 0.0f);
    for (size_t i = 0; i < W.size(); i++) {
        for (size_t j = 0; j < x.size(); j++) {
            result[i] += W[i][j] * x[j];
        }
    }
    return result;
}

Vec add(const Vec& a, const Vec& b) {
    Vec result(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Vec relu(const Vec& v) {
    Vec result(v.size(), 0.0f);
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = std::fmax(0.0f, v[i]);
    }
    return result;
}

void print_vec(const Vec& v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

int main(int argc, char* argv[]) {
    // example: 3 input features
    Vec x = {1.0f, 2.0f, 3.0f};
    
    // weights
    Mat W = {
        {0.1f, 0.2f, 0.3f},  // neuron 1 weights
        {0.4f, 0.5f, 0.6f},  // neuron 2 weights
    };

    // bias
    Vec b = {0.1f, -0.2f};

    Vec out = relu(add(matvec_mul(W, x), b));

    std::cout << "output:\n";
    print_vec(out);
}
