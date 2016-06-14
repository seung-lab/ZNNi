#include "dataprovider.hpp"

int main(int argc, char* argv[])
{
  h5vec3 fov(9, 109, 109);
  h5vec3 output_size(16, 256, 256);

  DataProvider dp(output_size, fov);
  bool ok = false;
  if (argc == 4)
    ok = dp.LoadHDF(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
  else {
    printf("Usage: dataprovider inputfile.h5 outputfile.h5 dataset\n");
    //ok = dp.LoadHDF("channel.h5", "channel_result.h5", "/main");
  }

  if (!ok) return -1;


  // The dataprovider created all necessary overlapping windows (dataspaces), we only need to iterate over them.
  for (auto it = dp.begin(); it != dp.end(); ++it) {

    // Read a 3D input window [1x1x24x364x364]
    znn::fwd::host_tensor<float, 5> input_patch = dp.ReadWindowData(*it);

    //
    // DO ZNNi MAGIC HERE: converting from 3D input [1x1x24x364x364] into smaller 4D output [1x3x16x256x256]
    //

    //--> This part only does some arbitrary transformations to test the hdf read/write code
    znn::fwd::host_tensor<float, 5> output_patch(1, 3, output_size.x(), output_size.y(), output_size.z());

    h5vec3 halffov((fov - 1) / 2);
    h5vec3 input(output_size + fov - 1);
    hsize_t w, inz, iny, inx, outz, outy, outx;

    for (w = 0; w < 3; ++w) {
      for (inz = halffov.z(), outz = 0; outz < output_size.z(); ++inz, ++outz) {
        for (iny = halffov.y(), outy = 0; outy < output_size.y(); ++iny, ++outy) {
          for (inx = halffov.x(), outx = 0; outx < output_size.x(); ++inx, ++outx) {
            if (w == 0)      output_patch[0][0][outx][outy][outz] = 1.f - input_patch[0][0][inx][iny][inz];
            else if (w == 1) output_patch[0][1][outx][outy][outz] = std::max(0.f,std::min(1.f, 2.f * input_patch[0][0][inx][iny][inz]));
            else if (w == 2) output_patch[0][2][outx][outy][outz] = input_patch[0][0][inx][iny][inz];
          }
        }
      }
    }
    //<-- This part only does some arbitrary transformations to test the hdf read/write code


    // Write a 4D output window [1x3x16x256x256] back
    dp.WriteWindowData(*it, output_patch);
  }

  return 0;
}
