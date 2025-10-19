#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

#include "FastNoiseLite.h"
#include "shader_maps.h"

#include "blur.cuh"
#include "cuda_hello.cuh"
#include "erosion.cuh"
#include "noise_generator_c.cuh" // cuda noise generator
#include "noise_generator_d.cuh" // new noise techiques

namespace nb = nanobind;

static void bind_hello(nb::module_ &m) {
    m.def("hello", []() { return "Hello from Python!"; });
}

static void bind_cuda_hello(nb::module_ &m) {
    m.def("cuda_hello", []() { cuda_hello(); });
}

static void bind_noise_generator_c(nb::module_ &m) {

    nb::class_<noise_generator_c::NoiseGeneratorC>(m, "NoiseGeneratorC")
        .def(nb::init<>())
        .def_rw("period", &noise_generator_c::NoiseGeneratorC::period)
        .def_rw("seed", &noise_generator_c::NoiseGeneratorC::seed)
        .def_rw("type", &noise_generator_c::NoiseGeneratorC::type)

        .def("fill", [](noise_generator_c::NoiseGeneratorC &self, nb::ndarray<float> arr) {

                if (arr.ndim() != 2)
    throw std::runtime_error("Expected a 2D float32 array");


        int h = arr.shape(0);
        int w = arr.shape(1);
        float *data = arr.data();

        self.fill(data, w, h); });
}

static void bind_noise_generator_d(nb::module_ &m) {

    auto ngd = nb::class_<noise_generator_d::NoiseGeneratorD>(m, "NoiseGeneratorD")
                   .def(nb::init<>())

#define X(TYPE, NAME, DEFAULT_VAL) \
    .def_prop_rw(#NAME, &noise_generator_d::NoiseGeneratorD::get_##NAME, &noise_generator_d::NoiseGeneratorD::set_##NAME)
                       NOISE_GENERATOR_D_PARAMETERS
#undef X

                   // nb::enum_<NoiseType>(m, "NoiseType")
                   //     .value("Hash", NoiseType::Hash)
                   //     .value("Gradient", NoiseType::Gradient)
                   //     .value("Simplex", NoiseType::Simplex)
                   //     .value("Cellular", NoiseType::Cellular)
                   //     .export_values();

                   .def("fill", [](noise_generator_d::NoiseGeneratorD &self, nb::ndarray<float> arr) {

                if (arr.ndim() != 2)
    throw std::runtime_error("Expected a 2D float32 array");


        int h = arr.shape(0);
        int w = arr.shape(1);
        float *data = arr.data();

        self.fill(data, w, h); });

    // Type enumerators
    nb::enum_<noise_generator_d::NoiseGeneratorD::Type>(ngd, "Type")

#define X(NAME) \
    .value(#NAME, noise_generator_d::NoiseGeneratorD::Type::NAME)
        NOISE_GENERATOR_D_TYPES
#undef X
            .export_values();
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

static void bind_shader_maps(nb::module_ &m) {
}

static void bind_blur(nb::module_ &m) {

    m.def("blur", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, float amount, bool wrap) {
        int h = arr.shape(0);
        int w = arr.shape(1);
        blur::blur(arr.data(), w, h, amount, wrap); }, nb::arg("arr"), nb::arg("amount"), nb::arg("wrap") = false);
}

// helpers to make python objects which is a bit convoluted from python
namespace python_helper {

// make a 2D numpy array, very akward from raw C++
nb::object get_2d_numpy_array(int h, int w) {
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
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Error in make_array_simple: ") + e.what());
    }
}

nb::object get_list_of_lists(int h, int w) {

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
}

static void bind_helpers(nb::module_ &m) {
    // EXAMPLE create a list of lists, we can't seem to make a numpy array though
    m.def("get_list_of_lists", [](int h, int w) {
        return get_list_of_lists(h, w);
    });

    m.def("get_2d_numpy_array", [](int h, int w) -> nb::object {
        return get_2d_numpy_array(h, w);
    });
}

} // namespace python_helper

NB_MODULE(cuda_hello, m) {

    bind_hello(m);
    bind_cuda_hello(m);
    bind_noise_generator_c(m);
    bind_noise_generator_d(m);
    bind_erosion(m);

    bind_shader_maps(m);
    bind_blur(m);

    python_helper::bind_helpers(m);
}
