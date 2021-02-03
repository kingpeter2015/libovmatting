#include "ovmatter_base_impl.hpp"

using namespace ovlib::matter;

std::map<std::string, NewInstancePt> CMatterFactory::dynCreateMap;
std::map<int, NewInstancePt> CMatterFactory::dynCreateMapType;
