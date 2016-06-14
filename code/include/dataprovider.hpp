#pragma once

#include <zi/vl/vl.hpp>
#include <znn/tensor/tensor.hpp>
#include <H5Cpp.h>
#include <string>
#include <vector>
#include <unordered_map>

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

  bool VerifyDatasetInfo();
  void CreateDataspaces();

public:
  typedef std::vector< hid_t >::iterator iterator;
  typedef std::vector< hid_t >::const_iterator const_iterator;

  DataProvider(const h5vec3 & outputsize, const h5vec3 & fov);
  ~DataProvider();

  iterator       begin()       { return dshandles_.begin();  }
  const_iterator begin() const { return dshandles_.cbegin(); }
  iterator       end()         { return dshandles_.end();    }
  const_iterator end()   const { return dshandles_.cend();   }
  size_t         size()  const { return dshandles_.size();   }

  bool LoadHDF(const std::string filename_input, const std::string filename_output, const std::string datasetname);
  znn::fwd::host_tensor<float, 5> ReadWindowData(hid_t dataspaceid);
  void WriteWindowData(hid_t ds, const znn::fwd::host_tensor<float, 5> & data);



};

