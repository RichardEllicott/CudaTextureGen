/*

👻 TEMPLATE_E 20251213

*/

#pragma once

#include "template_macro_undef.h"

// standard pattern to expand a define to a "string" (with the quote marks)
#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

#include "cuda_types.cuh"
#include <iostream>
#include <string> // not sure?
#include <typeindex>
#include <vector>

namespace template_e {

// testing making a derived class
class Base {

  protected:
    bool _device_configured = false;
    bool _device_allocated = false; // use or not, not sure but marking if the maps are uploaded

    // template called by xmacro (just to reduce code in xmacro)
    template <typename T>
    void set_par(T &field, const T &value) {
        if (field != value) {
            _device_configured = false;
            field = value;
        }
    }

    core::cuda::Stream stream; // will be allocated along with object

    dim3 block; // we now computer block in configure_device
    dim3 grid;  // and grid

    // virtual void configure_device() = 0;

    std::vector<core::cuda::DeviceArrayBase *> _device_array_n_ptrs; // store pointers to the DeviceArrayN's for reflection

    // needs to be a virtual function as we need to use a macro to fill _device_array_n_ptrs
    virtual std::vector<core::cuda::DeviceArrayBase *> get_device_array_n_ptrs() = 0;

  public:
    // optional override
    virtual void deallocate_device() {
        for (auto &ptr : get_device_array_n_ptrs()) {
            ptr->free_device();
        }
        _device_allocated = false;
    }

    // implemented in next object, uploads the pars and calculates grid/block
    virtual void configure_device() = 0;

    // allocate all the maps etc, if required
    virtual void allocate_device() = 0;

    // run the actual calculations
    virtual void process() = 0;

    // optional random array
    core::cuda::CurandArray2D curand_array_2d;
};

// ParametersT is the parameters struct
template <typename ParametersT>
class TemplateE : public Base {

  protected:
    // ParametersT pars;                               // local par
    // core::cuda::DeviceStruct<ParametersT> dev_pars; // device side pars

    core::cuda::SyncedDeviceStruct<ParametersT> pars;

    // call before launching a kernel to ensure pars are uploaded and block/grid calculated
    void configure_device() override {

        auto &host_pars = pars.host();

        auto _block = host_pars._block;
        auto width = host_pars._width;
        auto height = host_pars._height;

        if (!_device_configured) {
            block = dim3(_block, _block);
            grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            // dev_pars.upload(pars);
            pars.upload();
            _device_configured = true;
        }
    }

    // init function, but needs to be called in derived
    void initialize() {
        for (const auto ptr : get_device_array_n_ptrs()) {
            ptr->set_stream(stream.get());
        }
        curand_array_2d.set_stream(stream.get());
        pars.set_stream(stream.get());
    }
};

} // namespace template_d
