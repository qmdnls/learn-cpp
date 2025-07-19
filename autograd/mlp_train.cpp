#include "tensor.hpp"
#include <iomanip>
#include <iostream>
#include <random>

// random helper for params / data
static std::mt19937 rng(std::random_device{}());
float urand(float a, float b) { std::uniform_real_distribution<float> d(a,b); return d(rng); }

std::shared_ptr<Tensor> rand_param(const std::vector<size_t>& shape, float low=-0.5f, float high=0.5f) {
    auto t = std::make_shared<Tensor>(shape, 0.0f, true);
    for (auto& v : t->data) v = urand(low,high);
    return t;
}

struct Sample { float x1, x2; size_t y; };

std::vector<Sample> make_two_moons(size_t n_per_moon = 100, float noise = 0.1f) {
    std::normal_distribution<float> g(0.0f, noise);
    std::vector<Sample> out;
    out.reserve(2 * n_per_moon);

    for (size_t i = 0; i < n_per_moon; ++i) {
        float t = float(i) / n_per_moon * M_PI;     // angle 0..π
        // upper moon (class 0)
        out.push_back({ std::cos(t) + g(rng) ,  std::sin(t) + g(rng) , 0 });
        // lower moon (class 1)
        out.push_back({ std::cos(t) + 1.0f + g(rng) , -std::sin(t) - 0.5f + g(rng) , 1 });
    }
    return out;
}

struct MLP {
    size_t in_dim, hidden_dim, out_dim;
    std::shared_ptr<Tensor> W1, b1, W2, b2;

    MLP(size_t in_d, size_t h_d, size_t out_d)
        : in_dim(in_d), hidden_dim(h_d), out_dim(out_d) {
        W1 = rand_param({hidden_dim, in_dim});
        b1 = rand_param({hidden_dim, 1});
        W2 = rand_param({out_dim, hidden_dim});
        b2 = rand_param({out_dim, 1});
    }

    // forward pass returning probs and caches needed for loss/grad
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x,
                                    std::shared_ptr<Tensor>& logits_out) {
        auto h1   = relu(add(matmul(W1, x), b1));
        logits_out = add(matmul(W2, h1), b2);
        return softmax(logits_out);
    }

    // single SGD step, returns loss
    float train_step(const std::shared_ptr<Tensor>& x, size_t label, float lr) {
        // ----- forward -----
        std::shared_ptr<Tensor> logits;
        auto probs = forward(x, logits);
        float loss = cross_entropy(probs, label);

        // ----- seed grad for softmax‑Xent -----
        probs->grad.assign(probs->data.size(), 0.0f);
        probs->grad[label] = -1.0f / probs->data[label];

        // ----- backward -----
        probs->backward();

        // ----- SGD update -----
        auto sgd = [lr](std::shared_ptr<Tensor> p){
            for (size_t i = 0; i < p->data.size(); ++i) {
                p->data[i] -= lr * p->grad[i];
            }
            p->zero_grad();
        };
        sgd(W1); sgd(b1); sgd(W2); sgd(b2);

        return loss;
    }
};

int main() {
    const size_t in_dim = 2;
    const size_t hidden_dim = 32;
    const size_t out_dim = 2;

    MLP net(in_dim, hidden_dim, out_dim);

    // toy dataset: XOR in 2D
    //struct Example { float x1, x2; size_t y; } data[] = {
    //    {0,0, 0},
    //    {0,1, 1},
    //    {1,0, 1},
    //    {1,1, 0}
    //};
    
    auto data = make_two_moons(1000, 0.2f);

    const float lr = 0.1f;
    const size_t epochs = 200;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (const auto& ex : data) {
            auto x = std::make_shared<Tensor>(std::vector<size_t>{in_dim,1}, 0.0f, false);
            x->data[0] = ex.x1;
            x->data[1] = ex.x2;
            total_loss += net.train_step(x, ex.y, lr);
        }
        if ((epoch+1)%10==0) {
            std::cout << "epoch " << (epoch+1) << " loss: " << total_loss/4 << "\n";
        }
        std::shuffle(data.begin(), data.end(), rng);
    }

    // evaluate
    //std::cout << "\nafter training:\n";
    //for (const auto& ex : data) {
    //    auto x = std::make_shared<Tensor>(std::vector<size_t>{in_dim,1},0.0f,false);
    //    x->data[0]=ex.x1; x->data[1]=ex.x2;
    //    std::shared_ptr<Tensor> logits;
    //    auto probs = net.forward(x, logits);
    //    std::cout << ex.x1 << " " << ex.x2 << " -> p0=" << std::fixed << std::setprecision(3)
    //              << probs->data[0] << " p1=" << probs->data[1] << "\n";
    //}
}

