#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "core_api.h"

#include "c_noise_generator.cuh"

#include "erosion.cuh"

#include "FastNoiseLite.h"

#include <vector>

namespace nb = nanobind;

static void bind_hello(nb::module_ &m) {
    m.def("hello", []() { return "Hello from Python!"; });
}

static void bind_cuda_hello(nb::module_ &m) {
    m.def("cuda_hello", []() { cuda_hello(); });
}

static void bind_c_noise_generator(nb::module_ &m) {

    nb::class_<c_noise_generator::CNoiseGenerator>(m, "CNoiseGenerator")
        .def(nb::init<>())
        // .def_rw("scale", &c_noise_generator::CNoiseGenerator::scale)
        .def_rw("period", &c_noise_generator::CNoiseGenerator::period)
        .def_rw("seed", &c_noise_generator::CNoiseGenerator::seed)

        .def("fill", [](c_noise_generator::CNoiseGenerator &self, nb::ndarray<float> arr) {

                if (arr.ndim() != 2)
    throw std::runtime_error("Expected a 2D float32 array");


        int h = arr.shape(0);
        int w = arr.shape(1);
        float *data = arr.data();

        self.fill(data, w, h); });
}

static void bind_erosion(nb::module_ &m) {

    nb::class_<erosion::ErosionSimulator>(m, "ErosionSimulator")
        .def(nb::init<>())

        // .def_rw("rain_rate", &erosion::ErosionSimulator::rain_rate)
        // .def_rw("evaporation_rate", &erosion::ErosionSimulator::evaporation_rate)
        .def_rw("erosion_rate", &erosion::ErosionSimulator::erosion_rate)
        .def_rw("deposition_rate", &erosion::ErosionSimulator::deposition_rate)
        .def_rw("slope_threshold", &erosion::ErosionSimulator::slope_threshold)
        .def_rw("steps", &erosion::ErosionSimulator::steps)

// --------------------------------------------------------------------------------
// Declare CUDA constants
#define X(TYPE, NAME, DEFAULT_VAL) \
    .def_prop_rw(#NAME, &erosion::ErosionSimulator::get_##NAME, &erosion::ErosionSimulator::set_##NAME)
            EROSION_CONSTANTS
#undef X
        // --------------------------------------------------------------------------------

        // .def_prop_rw("max_height", &erosion::ErosionSimulator::get_max_height, &erosion::ErosionSimulator::set_max_height)
        // .def_prop_rw("min_height", &erosion::ErosionSimulator::get_min_height, &erosion::ErosionSimulator::set_min_height)

        .def("run_erosion", [](erosion::ErosionSimulator &self, nb::ndarray<float> arr) {
            if (arr.ndim() != 2)
                throw std::runtime_error("Input must be a 2D float32 array");

            int height = arr.shape(0);
            int width = arr.shape(1);
            float *data = arr.data();
            self.run_erosion(data, width, height);
        });
}

NB_MODULE(cuda_hello, m) {

    bind_hello(m);
    bind_cuda_hello(m);
    bind_c_noise_generator(m);
    bind_erosion(m);

    // // Erosion
    // m.def("erosion", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, int steps) {
    //         int h = arr.shape(0);
    //         int w = arr.shape(1);
    //         run_erosion(arr.data(), w, h, steps); }, nb::arg("arr"), nb::arg("steps"));

    // Blur
    m.def("blur", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, float amount, bool wrap) {
        int h = arr.shape(0);
        int w = arr.shape(1);
        run_blur(arr.data(), w, h, amount, wrap); }, nb::arg("arr"), nb::arg("amount"), nb::arg("wrap") = false);
    //
    //
    //
    // generate_noise

    m.def("generate_noise",
          [](int width, int height, float frequency) {
              return "test generate_noise return";
          });

    //
    //
    //

    // EXAMPLE create a list of lists, we can't seem to make a numpy array though
    m.def("make_list", [](int h, int w) {
        std::vector<nb::list> rows;
        rows.reserve(h);

        for (int y = 0; y < h; ++y) {
            nb::list row;
            for (int x = 0; x < w; ++x) {
                row.append(0.0f); // or some value
            }
            rows.push_back(row);
        }

        nb::list outer;
        for (auto &r : rows)
            outer.append(r);

        return outer; // Python will see a list of lists
    });

    //
    //
    //

    //
    // Claude got it!!!
    // https://claude.ai/chat/1b5b7537-a89b-4f07-8cb9-dd18b0557935
    //
    // Simple version that definitely works
    m.def("make_array_simple", [](int h, int w) -> nb::object {
    try {
        // Import numpy
        nb::module_ np = nb::module_::import_("numpy");
        
        // Create zeros array
        nb::object arr = np.attr("zeros")(nb::make_tuple(h, w));
        arr = arr.attr("astype")("float32");
        
        // Fill with values using Python indexing
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                arr[nb::make_tuple(y, x)] = static_cast<float>(y * w + x);
            }
        }
        
        return arr;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in make_array_simple: ") + e.what());
    } });

    //
    //
    //

    //
    //
    //
}
