#pragma once

#include "zi/vl/vl.hpp"
#include <znn/tensor/tensor.hpp>
#include <H5Cpp.h>
#include <string>
#include <vector>
#include <unordered_map>

#include <H5Exception.h>
#include <iterator>
#include <numeric>

namespace znn{ namespace fwd{

typedef zi::vl::vec<hsize_t, 3>  h5vec3;
typedef zi::vl::vec<hsize_t, 4>  h5vec4;

class DataProvider
{
private:
  typedef std::unordered_map< hid_t, std::pair< H5::DataSpace, H5::DataSpace > > DataSpaceMap;

  H5::H5File                  h5filein_;
  H5::H5File                  h5fileout_;
  H5::DataSet                 datasetin_;
  H5::DataSet                 datasetout_;

  H5T_class_t                 dataclass_;
  H5T_order_t                 dataorder_;
  H5T_sign_t                  datasign_;

  std::vector< hid_t >        dshandles_;  // DataSpace IDs for user
  DataSpaceMap                dsmap_;      // Mapping of DataSpace ID to input DataSpace (first) and output Dataspace (second)

  bool                        init_;
  const h5vec3                fov_;
  const h5vec3                outputsize_;
  const h5vec3                inputsize_;
  h5vec3                      world_;


public:
  typedef std::vector< hid_t >::iterator iterator;
  typedef std::vector< hid_t >::const_iterator const_iterator;

  iterator       begin()       { return dshandles_.begin();  }
  const_iterator begin() const { return dshandles_.cbegin(); }
  iterator       end()         { return dshandles_.end();    }
  const_iterator end()   const { return dshandles_.cend();   }
  size_t         size()  const { return dshandles_.size();   }

  DataProvider(const h5vec3 & outputsize, const h5vec3 & fov) :
    init_(false),
    fov_(fov),
    outputsize_(outputsize),
    inputsize_(outputsize + fov - 1),
    world_{ 0, 0, 0 }
  {
  }

  ~DataProvider()
  {
    for (auto it = dsmap_.begin(); it != dsmap_.end(); ++it) {
      it->second.first.close();
      it->second.second.close();
    }

    datasetin_.close();
    h5filein_.close();

    datasetout_.close();
    h5fileout_.close();
  }

  bool LoadHDF(const std::string filename_input, const std::string filename_output, const std::string datasetname)
  {
    try {
      h5filein_.openFile(filename_input.c_str(), H5F_ACC_RDONLY);
      datasetin_ = h5filein_.openDataSet(datasetname);

      if (!VerifyDatasetInfo()) {
        datasetin_.close();
        h5filein_.close();
        return false;
      }


      h5fileout_.setId(H5Fcreate(filename_output.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
      datasetout_ = h5fileout_.createDataSet(datasetname, H5::PredType::IEEE_F32LE, H5::DataSpace(4, h5vec4(3, world_).data()));

      CreateDataspaces();
      return true;
    }
    catch (H5::Exception e)
    {
      //e.printError();
      datasetin_.close();
      h5filein_.close();
      return false;
    }
    catch (...)
    {
      datasetin_.close();
      h5filein_.close();
      printf("Error: Unknown error while loading dataset '%s' in file '%s'.\n", datasetname.c_str(), filename_input.c_str());
      return false;
    }
  }

  bool VerifyDatasetInfo() {
    H5::DataSpace dataspace = datasetin_.getSpace();
    int dim_count = H5Sget_simple_extent_ndims(dataspace.getId());

    if (dim_count != 3) {
      printf("Error: Dataset should be rank 3, but is rank '%i'.\n", dim_count);
      return false;
    }

    h5vec3 max_size;
    H5Sget_simple_extent_dims(dataspace.getId(), world_.data(), max_size.data());
    printf("Dataset size : [%5llu, %5llu, %5llu]\n", world_[0], world_[1], world_[2]);
    printf("Field of view: [%5llu, %5llu, %5llu]\n", fov_[0], fov_[1], fov_[2]);
    printf("Inner window : [%5llu, %5llu, %5llu]\n", outputsize_[0], outputsize_[1], outputsize_[2]);
    printf("Outer window : [%5llu, %5llu, %5llu]\n", inputsize_[0], inputsize_[1], inputsize_[2]);

    if (world_.x() < outputsize_.x() || world_.y() < outputsize_.y() || world_.z() < outputsize_.z()) {
      printf("Error: Dataset is smaller than outer window size!\n");
      return false;
    }

    if ((fov_.x() % 2 == 0) || (fov_.y() % 2 == 0) || (fov_.z() % 2 == 0)) {
      printf("Error: Field of view must have an uneven number for each dimension");
      return false;
    }

    hid_t datatype = H5Dget_type(datasetin_.getId());
    dataclass_ = H5Tget_class(datatype);
    dataorder_ = H5Tget_order(datatype);
    datasign_ = H5Tget_sign(datatype);

    hsize_t datasize = H5Tget_size(datatype);
    printf("Datatype     : %llu Byte ", datasize);
    switch (datasign_) {
      case H5T_SGN_NONE: printf("UNSIGNED "); break;
      case H5T_SGN_2:    printf("SIGNED "); break;
      default:           break;
    }
    switch (dataorder_) {
      case H5T_ORDER_LE: printf("LITTLE ENDIAN "); break;
      case H5T_ORDER_BE: printf("BIG ENDIAN "); break;
      default:           break;
    }
    switch (dataclass_) {
      case H5T_INTEGER: printf("INTEGER\n"); break;
      case H5T_FLOAT:   printf("FLOAT\n"); break;
      default:          printf("UNKNOWN\n"); break;
    }

    if (! (datasize == 4 && dataclass_ == H5T_FLOAT) &&
        ! (datasize == 1 && datasign_ == H5T_SGN_NONE && dataclass_ == H5T_INTEGER )) {
      printf("Error: Datatype should be float or unsigned char.");
      return false;
    }

    dataspace.close();
    return true;
  }

  void CreateDataspaces()
  {
    h5vec3 windowcount(ceilf((world_[0] - fov_[0] + 1) / float(outputsize_[0])),
        ceilf((world_[1] - fov_[1] + 1) / float(outputsize_[1])),
        ceilf((world_[2] - fov_[2] + 1) / float(outputsize_[2])));

    h5vec3 halffov((fov_ - 1) / 2);
    h5vec3 start(0, 0, 0);
    h5vec3 end(world_);
    bool hitboundary_x, hitboundary_y, hitboundary_z;

    dsmap_.reserve(windowcount[0] * windowcount[1] * windowcount[2]);
    dshandles_.reserve(windowcount[0] * windowcount[1] * windowcount[2]);

    start.z() = 0; hitboundary_z = false;
    while (!hitboundary_z) {
      if (start.z() > world_.z() - inputsize_.z()) {      // zBoundary special case
        start.z() = world_.z() - inputsize_.z();
        hitboundary_z = true;
      }

      start.y() = 0; hitboundary_y = false;
      while (!hitboundary_y) {
        if (start.y() > world_.y() - inputsize_.y()) {    // yBoundary special case
          start.y() = world_.y() - inputsize_.y();
          hitboundary_y = true;
        }

        start.x() = 0; hitboundary_x = false;
        while (!hitboundary_x) {
          if (start.x() > world_.x() - inputsize_.x()) {  // xBoundary special case
            start.x() = world_.x() - inputsize_.x();
            hitboundary_x = true;
          }

          H5::DataSpace dsInput = datasetin_.getSpace();
          H5::DataSpace dsOutput = datasetout_.getSpace();

          H5Sselect_hyperslab(dsInput.getId(), H5S_SELECT_SET, start.data(), NULL, inputsize_.data(), NULL);
          H5Sselect_hyperslab(dsOutput.getId(), H5S_SELECT_SET, h5vec4(0, start + halffov).data(), NULL, h5vec4(3, outputsize_).data(), NULL);

          dshandles_.push_back(dsInput.getId());
          dsmap_.emplace(dsInput.getId(), std::make_pair(dsInput, dsOutput));

          start.x() += outputsize_.x();
        }
        start.y() += outputsize_.y();
      }
      start.z() += outputsize_.z();
    }
  }

  host_tensor<float, 5> ReadWindowData(hid_t dataspaceid)
  {
    hid_t memspace = H5Screate_simple(3, inputsize_.data(), NULL);

    host_tensor<float, 5> data_out(1, 1, inputsize_.x(), inputsize_.y(), inputsize_.z());
      H5Dread(datasetin_.getId(), H5T_NATIVE_FLOAT, memspace, dataspaceid, H5P_DEFAULT, data_out.ptr().get());

    if (dataclass_ == H5T_INTEGER) { // raw channel data (UINT8) needs to be normalized
      hsize_t elementcnt = inputsize_[1] * inputsize_[2];

      for (hsize_t z = 0; z < inputsize_[0]; ++z) {
        auto begin = data_out[0][0][z].begin();
        auto end = data_out[0][0][z].end();

        double sum = std::accumulate(begin, end, 0.0);
        double mean = sum / (double)elementcnt;

        std::vector<double> diff(elementcnt);
        std::transform(begin, end, diff.begin(), [mean](double x) { return x - mean; });

        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / (double)elementcnt);

        for (auto it = data_out[0][0][z].begin(); it != data_out[0][0][z].end(); ++it) {
          *it = (*it - (mean / stddev)) / 255.f;
        }
      }
    }

    H5Sclose(memspace);
    return data_out;
  }

  void WriteWindowData(hid_t dataspaceid, const host_tensor<float, 5> & data)
  {
    hid_t memspace = H5Screate_simple(4, h5vec4(3, outputsize_).data(), NULL);
    hid_t dataspace_out = dsmap_[dataspaceid].second.getId();

    h5vec4 start, end;
    H5Sget_select_bounds(dataspace_out, start.data(), end.data());
    printf("Writing data between [%5llu, %5llu, %5llu, %5llu] and [%5llu, %5llu, %5llu, %5llu]\n", start.x(), start.y(), start.z(), start.w(), end.x(), end.y(), end.z(), end.w());

    H5Dwrite(datasetout_.getId(), H5T_NATIVE_FLOAT, memspace, dataspace_out, H5P_DEFAULT, data.ptr().get());

    H5Sclose(memspace);
  }

};

  }} // namespace znn::fwd
