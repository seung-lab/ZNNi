#pragma once

#include <znn/types.hpp>
#include <znn/tensor/tensor.hpp>

namespace znn {
  namespace fwd {

    // Magic function that converts the ZNN kernels from
    // (in, out, a, b, c) to (out, in, c, b, a)
    // Host tensors only, overwrites input
    template< typename T >
    inline void fix_dims(T * inout,
      long_t n_in, long_t n_out,
      long_t a, long_t b, long_t c)
    {
      host_tensor<T, 5> cpy(n_in, n_out, c, b, a);
      cpy.load(inout, from_host);

      host_tensor_ref<T, 5> in_tensor(cpy.data(), n_in, n_out, c, b, a);
      host_tensor_ref<T, 5> out_tensor(inout, n_out, n_in, a, b, c);
      for (long_t i = 0; i < n_in; ++i)
        for (long_t j = 0; j < n_out; ++j)
          for (long_t k = 0; k < a; ++k)
            for (long_t l = 0; l < b; ++l)
              for (long_t m = 0; m < c; ++m)
                out_tensor[j][i][k][l][m] = in_tensor[i][j][m][l][k];
    }
  }
}