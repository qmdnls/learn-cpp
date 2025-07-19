#include "tensor.hpp"
#include <algorithm>        // std::max_element
#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_set>

//----------------------------------
// tensor constructors + utils
//----------------------------------

// helper to get element counts from shape
static size_t prod(const std::vector<size_t>& dims) {
    size_t p = 1;
    for (size_t d : dims) p *= d;
    return p;
}

// constructor
Tensor::Tensor(const std::vector<size_t>& shape_, float fill, bool req_grad) : shape(shape_), requires_grad(req_grad) {
    data.assign(prod(shape), fill);
    if (requires_grad) {
        grad.assign(data.size(), 0.0f);
    }
}

// zeroing grad buffer
void Tensor::zero_grad() {
    if (!grad.empty()) {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
}

// reverse‑mode autodiff
void Tensor::backward() {
    // initialize grad if empty
    if (grad.empty()) {
        grad.assign(data.size(), 0.0f);
    }
    
    // if grad is scalar set to 1
    if (grad.size() == 1 && grad[0] == 0.0f) {
        grad[0] = 1.0f;
    }

    // build topo ordering (via depth-first)
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> seen;
    
    // recursive lambda function for dfs
    auto build = [&](auto&& self, Tensor* t) -> void {
        if (seen.count(t)) {
            return;
        }
        seen.insert(t);
        for (auto& p : t->_prev) {
            self(self, p.get());
        }
        topo.push_back(t);
    };
    build(build, this);

    for (Tensor* t : topo) {
        if (t->requires_grad && t->grad.empty()) {
            t->grad.assign(t->data.size(), 0.0f);
        }
    }

    // walk in reverse order calling each backward lambda
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }}

// factory helper
std::shared_ptr<Tensor> tensor(
    const std::vector<float>& values,
    const std::vector<size_t>& shape,
    bool requires_grad
) {
    auto t = std::make_shared<Tensor>(shape, 0.0f, requires_grad);
    assert(values.size() == t->data.size());
    t->data = values;
    return t;
}

//----------------------------------
// basic ops
//----------------------------------

static void ensure_grad(std::shared_ptr<Tensor> t) {
    if (t->requires_grad && t->grad.empty()) {
        t->grad.assign(t->data.size(), 0.0f);
    }
}

std::shared_ptr<Tensor> add(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
) {
    assert(a->shape == b->shape);
    
    // output requires grad if either a or b requires grad
    auto out = std::make_shared<Tensor>(
        a->shape,
        0.0f,
        a->requires_grad || b->requires_grad
    );

    // perform add
    for (size_t i = 0; i < out->data.size(); ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }

    // save parents in graph
    out->_prev = {a, b};
    
    // define backward fn for each parent
    out->_backward = [a, b, out]() {
        if (a->requires_grad) {
            ensure_grad(a);
            for (size_t i = 0; i < a->grad.size(); ++i) {
                a->grad[i] += out->grad[i];
            }
        }
        if (b->requires_grad) {
            ensure_grad(b);
            for (size_t i = 0; i < b->grad.size(); ++i) {
                b->grad[i] += out->grad[i];
            }
        }
    };

    return out;
}

std::shared_ptr<Tensor> matmul(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
) {
    // assume 2‑D shapes: (m,n) * (n,p)
    assert(a->shape.size() == 2 && b->shape.size() == 2);
    size_t m = a->shape[0], n = a->shape[1], p = b->shape[1];
    assert(n == b->shape[0]);

    // output of shape (m,p)
    auto out = std::make_shared<Tensor>(
        std::vector<size_t>{m, p},
        0.0f,
        a->requires_grad || b->requires_grad
    );

    // perform matmul
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < n; ++k) {
            float a_val = (*a)(i, k);
            for (size_t j = 0; j < p; ++j) {
                (*out)(i, j) += a_val * (*b)(k, j);
            }
        }
    }

    // save parents in graph
    out->_prev = {a, b};
    
    // define backward fn for each parent
    if (a->requires_grad || b->requires_grad) {
        out->_backward = [a, b, out](void) {
            size_t m = a->shape[0], n = a->shape[1], p = b->shape[1];

            if (a->requires_grad) ensure_grad(a);
            if (b->requires_grad) ensure_grad(b);

            // dA = dC · B^T
            if (a->requires_grad) {
                for (size_t i = 0; i < m; ++i)
                    for (size_t k = 0; k < n; ++k) {
                        float grad_val = 0.0f;
                        for (size_t j = 0; j < p; ++j)
                            grad_val += out->grad[i * p + j] * (*b)(k, j);
                        a->grad[i * n + k] += grad_val;
                    }
            }

            // dB = A^T · dC
            if (b->requires_grad) {
                for (size_t k = 0; k < n; ++k) {
                    for (size_t j = 0; j < p; ++j) {
                        float grad_val = 0.0f;
                        for (size_t i = 0; i < m; ++i)
                            grad_val += (*a)(i, k) * out->grad[i * p + j];
                        b->grad[k * p + j] += grad_val;
                    }
                }
            }
        };
    } else {
        // if no parents need grads we can leave the default empty lambda
    } 
    return out;
}

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& x) {
    auto out = std::make_shared<Tensor>(x->shape, 0.0f, x->requires_grad);
    
    // perform relu
    for (size_t i = 0; i < x->data.size(); ++i) {
        out->data[i] = std::fmax(0.0f, x->data[i]);
    }

    // backward fn
    if (x->requires_grad) {
        out->_prev = {x};
        out->_backward = [x, out]() {
            ensure_grad(x);
            for (size_t i = 0; i < x->grad.size(); ++i)
                x->grad[i] += (x->data[i] > 0.0f ? out->grad[i] : 0.0f);
        };
    }
    
    return out;
}

std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& x, size_t /*dim*/) {
    // assume column vector shape (n,1)
    assert(x->shape.size() == 2 && x->shape[1] == 1);
    size_t n = x->shape[0];
    auto out = std::make_shared<Tensor>(x->shape, 0.0f, x->requires_grad);
    float max_val = *std::max_element(x->data.begin(), x->data.end());
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        out->data[i] = std::exp(x->data[i] - max_val);
        sum += out->data[i];
    }
    for (float &v : out->data) v /= sum;

    if (x->requires_grad) {
        out->_prev = {x};
        out->_backward = [x, out]() {
            ensure_grad(x);
            size_t n = x->shape[0];
            // jacobian‑vector product: grad_input = out * (grad ‑ sum(grad*out))
            float dot = 0.0f;
            for (size_t i = 0; i < n; ++i) dot += out->grad[i] * out->data[i];
            for (size_t i = 0; i < n; ++i)
                x->grad[i] += out->data[i] * (out->grad[i] - dot);
        };
    }
    return out;
}

float cross_entropy(const std::shared_ptr<Tensor>& probs, size_t target) {
    assert(probs->shape[1] == 1 && target < probs->shape[0]);
    return -std::log(std::max(probs->data[target], 1e-12f));
}
