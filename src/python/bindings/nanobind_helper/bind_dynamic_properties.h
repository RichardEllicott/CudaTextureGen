/*

bind the python __getattr__ and __setattr__ so we will first check for a registered nanobind property
and if not we will use fallback has,get,set functions

this allows us to store properties for example in a container like an unordered_map

================================================================================================================================
[Example Binding]
--------------------------------------------------------------------------------------------------------------------------------

nanobind::helper::bind_dynamic_properties<GNA_Base>(
    ngd,
    &GNA_Base::has_property,
    &GNA_Base::get_property,
    &GNA_Base::set_property);

================================================================================================================================
[Example Property Storage]
--------------------------------------------------------------------------------------------------------------------------------

std::unordered_map<std::string, nb::object> properties;

bool has_property(const std::string &key) const {
    return properties.find(key) != properties.end();
}

nb::object get_property(const std::string &key) const {
    auto it = properties.find(key);
    if (it == properties.end()) {
        throw nb::attribute_error((std::string("Attribute not found: ") + key).c_str());
    }
    return it->second;
}

void set_property(const std::string &key, nb::object value) {
    properties[key] = value;
}

*/
#pragma once
#include <nanobind/nanobind.h>

namespace nanobind::helper {

namespace nb = nanobind;

// bind dynamic properties, but note the container set will store nanobind object's
// therefore this pattern is entirely dependant on nanobind
template <typename T>
void bind_dynamic_properties(
    nb::class_<T> &cls,
    bool (T::*has_property)(const std::string &) const,       // has_property member function
    nb::object (T::*get_property)(const std::string &) const, // get_property member function
    void (T::*set_property)(const std::string &, nb::object)  // set_property member function

) {
    //
    // __getattr__
    //
    cls.def("__getattr__", [=](nb::handle self, const char *key) -> nb::object {
        nb::object key_obj = nb::str(key);

        // 1. Try Python/nanobind attributes first
        if (PyObject *result = PyObject_GenericGetAttr(self.ptr(), key_obj.ptr()))
            return nb::steal<nb::object>(result);

        PyErr_Clear();

        // 2. Try dynamic properties
        auto &cpp = nb::cast<T &>(self);
        if ((cpp.*has_property)(key))
            return (cpp.*get_property)(key);

        // 3. Not found
        PyErr_Format(PyExc_AttributeError, "No such attribute: %s", key);
        throw nb::python_error();
    });

    //
    // __setattr__
    //
    cls.def("__setattr__", [=](nb::handle self, const char *key, nb::object value) {
        nb::object key_obj = nb::str(key);

        // 1. Try Python/nanobind attributes first
        if (PyObject_GenericSetAttr(self.ptr(), key_obj.ptr(), value.ptr()) == 0)
            return;

        PyErr_Clear();

        // 2. Store dynamic property
        auto &cpp = nb::cast<T &>(self);
        (cpp.*set_property)(key, value);
    });
}

} // namespace nanobind::helper