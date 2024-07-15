#include "Python/Benne.hpp"
#include "Python/Birch.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace SESAME;
using namespace std;

PYBIND11_MODULE(pysame, m) {
  py::enum_<BenneObj>(m, "BenneObj")
      .value("balance", BenneObj::balance)
      .value("accuracy", BenneObj::accuracy)
      .value("efficiency", BenneObj::efficiency)
      .value("accuracy_no_migration", BenneObj::accuracy_no_migration)
      .export_values();
  py::class_<Benne>(m, "Benne")
      .def(py::init<int, int, int, double, int, int, double, int, double>(),
           py::arg("dim") = 2, py::arg("n_clusters") = 3, py::arg("obj") = 0,
           py::arg("distance_threshold") = 0.5,
           py::arg("queue_size_threshold") = 100, py::arg("dim_threshold") = 2,
           py::arg("variance_threshold") = 0.5,
           py::arg("outliers_num_threshold") = 10,
           py::arg("outliers_dist_threshold") = 0.5)
      .def("fit", &Benne::fit, py::return_value_policy::reference_internal)
      .def("fit_predict", &Benne::fit_predict)
      .def("get_params", &Benne::get_params)
      .def("partial_fit", &Benne::partial_fit,
           py::return_value_policy::reference_internal)
      .def("predict", &Benne::predict)
      .def("set_output", &Benne::set_output)
      .def("set_params", &Benne::set_params);
  py::class_<Birch>(m, "Birch")
      .def(py::init<int, int, double, int>(), py::arg("dim") = 2,
           py::arg("n_clusters") = 3, py::arg("distance_threshold") = 0.5,
           py::arg("branching_factor") = 50)
      .def("fit", &Birch::fit, py::return_value_policy::reference_internal)
      .def("fit_predict", &Birch::fit_predict)
      .def("get_params", &Birch::get_params)
      .def("partial_fit", &Birch::partial_fit,
           py::return_value_policy::reference_internal)
      .def("predict", &Birch::predict)
      .def("set_output", &Birch::set_output)
      .def("set_params", &Birch::set_params);
}