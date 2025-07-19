#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

using Vec = std::vector<float>; // single float vector
using Mat = std::vector<Vec>; // row-major Matrix

std::mt19937 rng(std::random_device{}());  // seed once globally

float randf(float low, float high) {
    std::uniform_real_distribution<float> uniform(low, high);
    return uniform(rng);
}

Vec random_vec(size_t size, float low = -1.0f, float high = 1.0f) {
    Vec v(size);
    for (float& val : v) {
        val = randf(low, high);
    }
    return v;
}

Mat random_mat(size_t rows, size_t cols, float low = -1.0f, float high = 1.0f) {
    Mat m(rows);
    for (size_t i = 0; i < rows; ++i) {
        m[i] = random_vec(cols, low, high);
    }
    return m;
}

Vec he_init_vec(size_t fan_in, size_t size) {
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));
    Vec v(size);
    for (float& val : v) {
        val = dist(rng);
    }
    return v;
}

Mat he_init_mat(size_t rows, size_t cols) {
    Mat m(rows);
    for (size_t i = 0; i < rows; ++i) {
        m[i] = he_init_vec(cols, cols);
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

Vec softmax(const Vec& v) {
    Vec result(v.size(), 0.0f);
    float max_val = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = std::exp(v[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < result.size(); i++) {
        result[i] /= sum;
    }
    return result;
}

void print_vec(const Vec& v) {
    std::cout << "[";
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

    std::cout << "]\n";
}

void print_mat(const Mat& m) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[\n";
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
    std::cout << "]\n";
}


int main(int argc, char* argv[]) {
    size_t in_dim = 3;
    size_t hidden_dim = 4;
    size_t out_dim = 2;
    
    // example: 3 input features
    Vec x = random_vec(in_dim);;

    // untransposed weights of shape out_dim x in_dim + biases
    Mat w1 = he_init_mat(hidden_dim, in_dim);
    Vec b1 = random_vec(hidden_dim);

    Mat w2 = he_init_mat(out_dim, hidden_dim);
    Vec b2 = random_vec(out_dim);

    // forward
    Vec h1 = relu(add(matvec_mul(w1, x), b1));
    Vec out = softmax(add(matvec_mul(w2, h1), b2));

    std::cout << "input:\n";
    print_vec(x);
    std::cout << "\n";
    
    std::cout << "output:\n";
    print_vec(out);
}
