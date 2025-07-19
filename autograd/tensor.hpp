#pragma once            // include‑guard (similar to #ifndef)

#include <cassert>
#include <cstddef>      // size_t
#include <functional>   // std::function for custom backward lambdas
#include <memory>       // we use shared_ptr for computation graph 
#include <vector>       // dynamic arrays (fast implementation should probably use fixed-size)

struct Tensor {
    // raw flattened row-major data
    std::vector<float> data;
    std::vector<size_t> shape; // e.g. {rows, cols}

    // gradient storage (same length as data when requires_grad=true)
    std::vector<float> grad;

    // whether to track with autograd
    bool requires_grad = false;

    std::function<void()> _backward = []{}; // local backprop closure
    std::vector<std::shared_ptr<Tensor>> _prev; // parents in the computation graph

    // constructors
    Tensor() = default; // default: empty tensor
    Tensor(
        const std::vector<size_t>& shape,
        float fill_value = 0.0f,
        bool requires_grad = false
    );

    // helpers
    size_t numel() const { return data.size(); }
    void zero_grad(); // reset grad buffer to zeros
    void backward(); // launch reverse‑mode autodiff
    
    // operator overload for 2D element access on flattened data
    inline float& operator()(size_t r, size_t c) {
        assert(shape.size() == 2);
        return data[r * shape[1] + c];
    }
    inline float operator()(size_t r, size_t c) const {
        assert(shape.size() == 2);
        return data[r * shape[1] + c];
    }
};

// create a tensor from explicit values
std::shared_ptr<Tensor> tensor(
    const std::vector<float>& values,
    const std::vector<size_t>& shape,
    bool requires_grad = false
);

// basic ops

// each op constructs a new tensor, wires _prev / _backward so that calling
// .backward() on the output computes gradients for any parent that had
// requires_grad=true.

std::shared_ptr<Tensor> add(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
);

std::shared_ptr<Tensor> matmul(
    const std::shared_ptr<Tensor>& a,
    const std::shared_ptr<Tensor>& b
);

std::shared_ptr<Tensor> relu(
    const std::shared_ptr<Tensor>& x
);

std::shared_ptr<Tensor> softmax(
    const std::shared_ptr<Tensor>& x,
    size_t dim = 0
); // dim currently ignored (1‑D)

float cross_entropy(
    const std::shared_ptr<Tensor>& probs,
    size_t target_class
);
