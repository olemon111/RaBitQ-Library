#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <rabitqlib/index/symqg/qg.hpp>
#include <vector>
#include <cstdint>

namespace py = pybind11;
using namespace rabitqlib::symqg;

// Python binding for QuantizedGraph
PYBIND11_MODULE(_qg_core, m) {
    m.doc() = "RaBitQ QuantizedGraph Python bindings";

    py::class_<QuantizedGraph<float>>(m, "QuantizedGraphCore")
        .def(py::init<>())
        .def("load_index", [](QuantizedGraph<float> &self, const std::string &meta_filename, const std::string &data_filename) {
            self.load_index(meta_filename.c_str(), data_filename.c_str());
        }, "Load index from metadata and data files",
        py::arg("meta_filename"), py::arg("data_filename"))
        .def("set_ef", &QuantizedGraph<float>::set_ef, "Set ef parameter",
        py::arg("ef"))
        .def("search", [](QuantizedGraph<float> &self, 
                          py::array_t<float> query, 
                          uint32_t k) -> py::array_t<uint32_t> {
            py::buffer_info query_buf = query.request();
            if (query_buf.ndim != 1) {
                throw std::runtime_error("Query must be 1-dimensional");
            }
            
            size_t dim = query_buf.size;
            const float *query_ptr = static_cast<const float *>(query_buf.ptr);
            
            std::vector<uint32_t> results(k);
            self.search(query_ptr, k, results.data());
            
            // Convert to numpy array
            py::array_t<uint32_t> result_array(k);
            py::buffer_info result_buf = result_array.request();
            uint32_t *result_ptr = static_cast<uint32_t *>(result_buf.ptr);
            std::copy(results.begin(), results.end(), result_ptr);
            
            return result_array;
        }, "Search for k nearest neighbors",
        py::arg("query"), py::arg("k"))
        .def("num_vertices", &QuantizedGraph<float>::num_vertices, "Get number of vertices")
        .def("dimension", &QuantizedGraph<float>::dimension, "Get dimension")
        .def("degree_bound", &QuantizedGraph<float>::degree_bound, "Get degree bound")
        .def("entry_point", &QuantizedGraph<float>::entry_point, "Get entry point");
}

