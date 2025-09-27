#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "core_api.h"
#include "erosion.h" // or better: declare in erosion.h

#include "FastNoiseLite.h"

#include <vector>

namespace nb = nanobind;

NB_MODULE(cuda_hello, m)
{
      m.def("hello", []()
            { return "Hello from Python!"; });

      m.def("cuda_hello", []()
            { cuda_hello(); });

      // Erosion
      m.def("erosion", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, int steps)
            {
            int h = arr.shape(0);
            int w = arr.shape(1);
            run_erosion(arr.data(), w, h, steps); }, nb::arg("arr"), nb::arg("steps"));

      // Blur
      m.def("blur", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, float amount, bool wrap)
            {
        int h = arr.shape(0);
        int w = arr.shape(1);
        run_blur(arr.data(), w, h, amount, wrap); }, nb::arg("arr"), nb::arg("amount"), nb::arg("wrap") = false);
      //
      //
      //
      // generate_noise

      m.def("generate_noise",
            [](int width, int height, float frequency)
            {
                  return "test generate_noise return";
            });

      //
      //
      //

      // EXAMPLE create a list of lists, we can't seem to make a numpy array though
      m.def("make_list", [](int h, int w)
            {
                  std::vector<nb::list> rows;
                  rows.reserve(h);

                  for (int y = 0; y < h; ++y)
                  {
                        nb::list row;
                        for (int x = 0; x < w; ++x)
                        {
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
      m.def("make_array_simple", [](int h, int w) -> nb::object
            {
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
