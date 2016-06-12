#include <fstream>
#include <sstream>
#include <iostream>

namespace znn { namespace fwd{

template <typename T>
inline bool
read_from_file( const std::string& fname, T* data, std::size_t n)
{
  FILE* f = std::fopen(fname.c_str(), "rbXS");
  if( !f ) return false;

  std::size_t nread = std::fread(data, sizeof(T), n, f);
  std::fclose(f);

  std::cout << "NREAD: " << nread << std::endl;

  return nread == n;
}

template < typename T >
inline bool
write_to_file( const std::string& fname,
               const T* data, std::size_t n)
{
  std::ofstream f(fname.c_str(), (std::ios::out | std::ios::binary) );
  if (!f) return false;

  f.write( reinterpret_cast<const char*>(data), n*sizeof(T) );
  return true;
}

  }} // namespace znn::fwd
