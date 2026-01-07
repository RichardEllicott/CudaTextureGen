/*


*/
#pragma once

#include <nanobind/nanobind.h>
#include <string>
#include <unordered_map>

// #include "core.h"
#include "core/cuda/types_collection.cuh"

class GNA_Base {

    std::unordered_map<std::string, nanobind::object> properties; // store of python properties

  public:
    // has a key
    bool has_property(const std::string &key) const {
        return properties.find(key) != properties.end();
    }

    // get a property
    nanobind::object get_property(const std::string &key) const {
        auto it = properties.find(key);
        if (it == properties.end()) {
            throw nanobind::attribute_error((std::string("Attribute not found: ") + key).c_str());
        }
        return it->second;
    }

    // set a property
    void set_property(const std::string &key, nanobind::object value) {
        properties[key] = value;
    }

    // get property or default
    template <typename T>
    T get_property_or_default(const std::string &key, const T &default_value) const {
        auto it = properties.find(key);
        if (it == properties.end()) {
            return default_value;
        }
        return nanobind::cast<T>(it->second);
    }

    core::cuda::Stream stream;

    virtual void process() = 0;
};
