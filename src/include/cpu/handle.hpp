#pragma once

#include <zi/utility/singleton.hpp>
#include "utils/task_package.hpp"

namespace znn { namespace fwd { namespace cpu {

namespace {
task_package& handle =
    zi::singleton<task_package>::instance();
}

}}} // namespace znn::fwd::cpu
