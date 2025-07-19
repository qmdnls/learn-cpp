#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

using Vec = std::vector<float>; // single float vector
using Mat = std::vector<Vec>; // row-major Matrix

std::mt19937 rng(std::random_device{}());  // seed once globally
std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

float randf() {
    return uniform(rng);
}

Vec random_vec(size_t size) {
    Vec v(size);
    for (float& val : v) {
        val = randf();
    }
    return v;
}

Mat random_mat(size_t rows, size_t cols) {
    Mat m(rows);
    for (size_t i = 0; i < rows; ++i) {
        m[i] = random_vec(cols);
    }
    return m;
}

Vec matvec_mul(const Mat& W, const Vec& x) {
    Vec result(W.size(), 0.0f);
    for (size_t i = 0; i < W.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            result[i] += W[i][j] * x[j];
        }
    }
    return result;
}

Vec add(const Vec& a, const Vec& b) {
    Vec result(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Vec relu(const Vec& v) {
    Vec result(v.size(), 0.0f);
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = std::fmax(0.0f, v[i]);
    }
    return result;
}

void print_vec(const Vec& v) {
    std::cout << "vec([";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < v.size(); ++i) {
        float val = v[i];
        if (val >= 0) {
            std::cout << " ";
        }
        std::cout << val;
        if (i + 1 < v.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "])\n";
}

void print_mat(const Mat& m) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "mat([\n";
    for (size_t i = 0; i < m.size(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < m[i].size(); ++j) {
            float val = m[i][j];
            if (val >= 0) {
                std::cout << " "; // pad positives
            }
            std::cout << val;
            if (j + 1 < m[i].size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i + 1 < m.size()) {
            std::cout << ",\n";
        } else {
            std::cout << "\n";
        }
    }
    std::cout << "])\n";
}


int main(int argc, char* argv[]) {
    size_t in_dim = 3;
    size_t out_dim = 2;
    
    // example: 3 input features
    Vec x = {1.0f, 2.0f, 3.0f};

    // weights
    Mat W = random_mat(out_dim, in_dim);  // shape: out_dim x in_dim
    
    // bias
    Vec b = random_vec(out_dim); // shape: out_dim

    // forward
    Vec out = add(matvec_mul(W, x), b);

    std::cout << "input:\n";
    print_vec(x);
    std::cout << "\n";
    
    std::cout << "W:\n";
    print_mat(W);
    std::cout << "\n";
    
    std::cout << "b:\n";
    print_vec(b);
    std::cout << "\n";

    std::cout << "output:\n";
    print_vec(out);
}
