/*

base class for a type of generic object with generic storage

*/
#pragma once

#define GNB_BASE_GENERIC_TYPE 0 // 0 = any, 1 = variant

#include "core.h"
#include "core/cuda/types.cuh"
#include <string>
#include <unordered_map>

#if GNB_BASE_GENERIC_TYPE == 0
#include <any>
#elif GNB_BASE_GENERIC_TYPE == 1
#include <variant>
#endif

// void set_property_from_python(GNB_Base* self, const std::string& key, nb::object value) {
//     if (nb::isinstance<nb::int_>(value)) {
//         self->set_property(key, value.cast<int>());
//     } else if (nb::isinstance<nb::float_>(value)) {
//         self->set_property(key, value.cast<float>());
//     } else if (nb::isinstance<nb::bool_>(value)) {
//         self->set_property(key, value.cast<bool>());
//     } else {
//         throw nb::type_error("Unsupported property type");
//     }
// }

// has problem atm that the nanobind objects don't get cast

#if GNB_BASE_GENERIC_TYPE == 0
using PropertyValue = std::any;
#elif GNB_BASE_GENERIC_TYPE == 1
using PropertyValue = std::variant<
    int,
    float,
    bool,
    std::string>;
#endif

class GNB_Base {
  protected:
    std::unordered_map<std::string, PropertyValue> properties;

  public:
    // check if key exists
    bool has_property(const std::string &key) const {
        return properties.find(key) != properties.end();
    }

    // get a property (throws if wrong type)
    template <typename T>
    T get_property(const std::string &key) const {
        auto it = properties.find(key);
        if (it == properties.end()) {
            throw std::runtime_error("Property not found: " + key);
        }

#if GNB_BASE_GENERIC_TYPE == 0
        std::cout << "Stored type for " << key << ": " << it->second.type().name() << "\n";
        return std::any_cast<T>(it->second);
#elif GNB_BASE_GENERIC_TYPE == 1
        return std::get<T>(it->second);
#endif
    }

    // set a property
    template <typename T>
    void set_property(const std::string &key, T value) {
        properties[key] = std::move(value);
    }

    // get property or default
    template <typename T>
    T get_property_or_default(const std::string &key, const T &default_value) const {
        auto it = properties.find(key);
        if (it == properties.end()) {
            return default_value;
        }

#if GNB_BASE_GENERIC_TYPE == 0
        return std::any_cast<T>(it->second);
#elif GNB_BASE_GENERIC_TYPE == 1
        return std::get<T>(it->second);
#endif
    }

    //
    //

#if GNB_BASE_GENERIC_TYPE == 0
    // explicit any-based API for bindings
    void set_property_any(const std::string &key, std::any v) {
        properties[key] = std::move(v);
    }

    const std::any &get_property_any(const std::string &key) const {
        auto it = properties.find(key);
        if (it == properties.end())
            throw std::runtime_error("Property not found: " + key);
        return it->second;
    }
#endif

  

    core::Ref<core::cuda::Stream> stream;

    virtual void process() = 0;

    GNB_Base() {
        stream.instantiate();
    }
};
