#pragma once
#include <znn/tensor/tensor.hpp>

namespace znn {	namespace fwd {


/* host_tensor<float, 5> center_crop(const host_tensor<float, 5> & data, zi::vl::vec5i & size)
 * 
 * data [in]  The input tensor
 * size [in]  The size of the output (might be changed if one dimension is larger than input tensor)
 *
 * return     The output tensor. Shape might differ from `size`, if `size` was larger than `data`
 */
host_tensor<float, 5> center_crop(const host_tensor<float, 5> & data, const zi::vl::vec5ll & cropsize) {
  STRONG_ASSERT((data.shape()[0] % 2 == cropsize[0] % 2) && (data.shape()[1] % 2 == cropsize[1] % 2) && (data.shape()[2] % 2 == cropsize[2] % 2) &&
                (data.shape()[3] % 2 == cropsize[3] % 2) && (data.shape()[4] % 2 == cropsize[4] % 2) &&
                (cropsize.min() > 0));

  
  zi::vl::vec5ll size;
  // If cropsize is larger than data, cropsize will be limited to data
  for (int i = 0; i < 5; ++i) {
    size[i] = std::min(data.shape_vec()[i], cropsize[i]);
  }

  host_tensor<float, 5> out(size);
  zi::vl::vec5ll start((data.size() - size) / 2);

  for (size_t v = start[0]; v < start[0] + size[0]; ++v) {
    for (size_t w = start[1]; w < start[1] + size[1]; ++w) {
      for (size_t x = start[2]; x < start[2] + size[2]; ++x) {
        for (size_t y = start[3]; y < start[3] + size[3]; ++y) {
          out[v - start[0]][w - start[1]][x - start[2]][y - start[3]].load_n(&(data[v][w][x][y][start[4]]), size[4], from_host);
        }
      }
    }
  }

  return out;
}


}}