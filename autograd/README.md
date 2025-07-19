# learn‑cpp

Random C++ explorations and experiments while learning the language. Each dir is self‑contained and uses only clang + make.

## Programs 

* **simple_grep** – small `grep` clone in modern c++. reads a file, prints matching lines. starter for i/o, `std::string`, and command‑line args.

* **simple_mlp** – forward‑only 2‑layer mlp (vectors + plain loops). no autograd, just matrices, relu, softmax.

* **autograd** – home‑rolled tensor class with reverse‑mode autodiff and a tiny two‑layer net that trains on xor / two‑moons. calls openblas for fast `gemm`.
