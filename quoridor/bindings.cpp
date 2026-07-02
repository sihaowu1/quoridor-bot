// pybind11 bindings for the Quoridor engine.
//
// Build with quoridor/build_ext.py, which drops the compiled module at
// alphazero/quoridor_engine.<abi>.so so it imports as
// ``alphazero.quoridor_engine``.  The Python-facing env wrapper lives in
// alphazero/quoridor_cpp.py; the semantics contract with the reference
// implementation is documented in quoridor.h and enforced by
// alphazero/test_cpp_backend.py.

#include <cstring>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quoridor.h"

namespace py = pybind11;
using quoridor::Engine;

PYBIND11_MODULE(quoridor_engine, m) {
    m.doc() = "C++ Quoridor engine (see alphazero/quoridor_cpp.py for the "
              "gymnasium-style wrapper)";

    m.def("num_actions", &quoridor::num_actions, py::arg("n"));
    m.def("obs_size", &quoridor::obs_size, py::arg("n"));

    py::class_<Engine>(m, "Engine")
        .def(py::init<int, int, int>(), py::arg("board_size") = 5,
             py::arg("walls") = 3, py::arg("max_moves") = 0)
        .def("reset", &Engine::reset)
        // (reward, terminated, truncated); observation/info are separate
        // calls so the wrapper controls what gets materialised.
        .def("step",
             [](Engine& e, int action) {
                 auto out = e.step(action);
                 return py::make_tuple(out.reward, out.terminated,
                                       out.truncated);
             },
             py::arg("action"))
        .def("legal_actions",
             [](const Engine& e) { return e.legal_actions(); })
        .def("observation",
             [](const Engine& e) {
                 auto v = e.observation();
                 py::array_t<double> arr(static_cast<py::ssize_t>(v.size()));
                 std::memcpy(arr.mutable_data(), v.data(),
                             v.size() * sizeof(double));
                 return arr;
             })
        .def("set_state", &Engine::set_state, py::arg("p1"), py::arg("p2"),
             py::arg("to_play"), py::arg("h_walls"), py::arg("v_walls"),
             py::arg("walls_p1"), py::arg("walls_p2"))
        .def("action_for_move", &Engine::action_for_move, py::arg("r"),
             py::arg("c"))
        .def("action_for_wall", &Engine::action_for_wall, py::arg("r"),
             py::arg("c"), py::arg("orientation"))
        .def("clone", [](const Engine& e) { return Engine(e); })
        .def("__copy__", [](const Engine& e) { return Engine(e); })
        .def("__deepcopy__",
             [](const Engine& e, py::dict) { return Engine(e); },
             py::arg("memo"))
        .def_property_readonly("board_size", &Engine::board_size)
        .def_property_readonly("max_walls", &Engine::max_walls)
        .def_property_readonly("max_moves", &Engine::max_moves)
        .def_property_readonly("num_actions", &Engine::action_count)
        .def_property_readonly("obs_dim", &Engine::obs_dim)
        .def_property_readonly("to_play", &Engine::to_play)
        .def_property_readonly("done", &Engine::done)
        .def_property_readonly("winner", &Engine::winner)
        .def_property_readonly("move_count", &Engine::move_count)
        .def("walls_left", &Engine::walls_left, py::arg("player"))
        .def("pawn", &Engine::pawn, py::arg("player"))
        .def("h_walls", &Engine::h_walls)
        .def("v_walls", &Engine::v_walls)
        .def("__str__", &Engine::to_string);
}
