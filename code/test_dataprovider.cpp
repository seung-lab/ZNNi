#include "dataprovider.hpp"

int main(int argc, char* argv[])
{
  h5vec3 fov(9, 109, 109);
  h5vec3 output(16, 256, 256);

  DataProvider dp(output, fov);
  bool ok = false;
  if (argc == 4)
    ok = dp.LoadHDF(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
  else {
    printf("Usage: dataprovider inputfile.h5 outputfile.h5 dataset\n");
    //ok = dp.LoadHDF("channel.h5", "channel_result.h5", "/main");
  }

  if (!ok) return -1;


  // CRYPTIC TEST - Iterate over all windows, do some nonsense transformations and write it back to the new file.
  for (auto it = dp.begin(); it != dp.end(); ++it) {
    h5vec3 dimensions;
    std::unique_ptr<float[]> data = dp.ReadWindowData(*it, dimensions);

    hsize_t elcnt = output.x() * output.y() * output.z();
    float * conv = new float[3 * elcnt];
    h5vec3 halffov((fov - 1) / 2);
    h5vec3 input(output + fov - 1);

    hsize_t w, inz, iny, inx, outz, outy, outx;

    for (w = 0; w < 3; ++w) {
      for (inz = halffov.z(), outz = 0; inz + halffov.z() < input.z(); ++inz, ++outz) {
        for (iny = halffov.y(), outy = 0; iny + halffov.y() < input.y(); ++iny, ++outy) {
          for (inx = halffov.x(), outx = 0; inx + halffov.x() < input.x(); ++inx, ++outx) {
            if (w == 0)      conv[0*elcnt + (outz + output.z() * (outy + output.y() * outx))] = 1.f - data.get()[(inz + input.z() * (iny + input.y() * inx))];
            else if (w == 1) conv[1*elcnt + (outz + output.z() * (outy + output.y() * outx))] = std::max(0.f,std::min(1.f, 2.f * data.get()[(inz + input.z() * (iny + input.y() * inx))]));
            else if (w == 2) conv[2*elcnt + (outz + output.z() * (outy + output.y() * outx))] = data.get()[(inz + input.z() * (iny + input.y() * inx))];
          }
        }
      }
    }

    dp.WriteWindowData(*it, conv);
    delete[] conv;
    data.reset();
  }

  return 0;
}
