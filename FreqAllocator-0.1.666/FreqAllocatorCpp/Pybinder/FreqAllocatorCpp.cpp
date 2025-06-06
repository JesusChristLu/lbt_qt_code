#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"

#include "single_qubit_model.h"
#include "two_qubit_model.h"
using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(FreqAllocatorCpp, m)
{
	m.doc() = "[Module FreqAllocatorCpp]";

	py::class_<ChipQubit>(m, "ChipQubit")
		.def_readonly("used", &ChipQubit::used)
		.def_readonly("x", &ChipQubit::x)
		.def_readonly("y", &ChipQubit::y)
		.def_readonly("frequency", &ChipQubit::frequency)
		.def_readonly("t_sq", &ChipQubit::t_sq)
		.def_readonly("anharm", &ChipQubit::anharm)
		// .def_readonly("sweet_point", &ChipQubit::sweet_point)
		.def_readonly("allow_freq", &ChipQubit::allow_freq)
		.def_readonly("allow_freq_int", &ChipQubit::allow_freq_int)
		.def_readonly("ac_spectrum", &ChipQubit::ac_spectrum)
		.def_readonly("t1_freq", &ChipQubit::t1_freq)
		.def_readonly("t1_t1", &ChipQubit::t1_t1)
		.def_readonly("xy_crosstalk", &ChipQubit::xy_crosstalk)
		.def_readonly("model", &ChipQubit::model)
		.def("assign_frequency", &ChipQubit::assign_frequency)
		.def("assign_frequency_on_range", &ChipQubit::assign_frequency_on_range)
		;

	py::class_<ChipCoupler>(m, "ChipCoupler")
		.def_readonly("used", &ChipCoupler::used)
		.def_readonly("qubit1", &ChipCoupler::qubit1)
		.def_readonly("qubit2", &ChipCoupler::qubit2)
		.def_readonly("coupler_id", &ChipCoupler::coupler_id)
		.def_readonly("neighbor_couplers", &ChipCoupler::neighbor_couplers)
		.def_readonly("t_twoq", &ChipCoupler::t_twoq)
		.def_readonly("frequency", &ChipCoupler::qubit1)
		.def_readonly("model", &ChipCoupler::model)
		.def("assign_frequency", &ChipCoupler::assign_frequency)
		;

	py::class_<ChipError>(m, "ChipError")
		.def(py::init<int, int>())
		.def_readonly("H", &ChipError::H)
		.def_readonly("W", &ChipError::W)
		.def_readonly("qubits", &ChipError::qubits)
		.def_readonly("n_available_qubits", &ChipError::n_available_qubits)
		.def_readonly("xy_crosstalk", &ChipError::xy_crosstalk)
		.def_readonly("error_arr", &ChipError::error_arr)
		.def_readonly("alpha_list", &ChipError::alpha_list)
		.def_readonly("mu_list", &ChipError::mu_list)
		.def_readonly("detune_list", &ChipError::detune_list)

		.def("qubit_name_idx", &ChipError::qubit_name_idx)
		.def("qubit_idx", &ChipError::qubit_idx)
		.def("qubit_idx_to_pos", &ChipError::qubit_idx_to_pos)
		.def("check_available_qubit_pos", &ChipError::check_available_qubit_pos)
		.def("get_neighbors", &ChipError::get_neighbors)
		.def("get_neighbors_distance_sqrt2", &ChipError::get_neighbors_distance_sqrt2)
		.def("get_neighbors_distance_2", &ChipError::get_neighbors_distance_2)
		.def("get_distance", &ChipError::get_distance)

		.def("load_file", &ChipError::load_file)
		.def("_read_file_to_json", &ChipError::_read_file_to_json)
		.def("_load_qubit_data", &ChipError::_load_qubit_data)
		.def("_load_xy_crosstalk_sim_data", &ChipError::_load_xy_crosstalk_sim_data)
		.def("assemble_nodes", &ChipError::assemble_nodes)
		.def("_assemble_node", &ChipError::_assemble_node)

		.def("from_qubits_to_coupler_idx", &ChipError::from_qubits_to_coupler_idx)
		.def("from_coupler_idx_to_qubits", &ChipError::from_coupler_idx_to_qubits)
		.def("check_available_coupler_pos", &ChipError::check_available_coupler_pos)
		.def("coupler_pos_to_coupler_id", &ChipError::coupler_pos_to_coupler_id)
		.def("get_neighbor_couplers", &ChipError::get_neighbor_couplers)
		.def("assemble_couplers", &ChipError::assemble_couplers)
		.def("_assemble_coupler", &ChipError::_assemble_coupler)

		.def("initialize_all_qubits", &ChipError::initialize_all_qubits)
		.def("list_all_unallocated_qubits", &ChipError::list_all_unallocated_qubits)
		.def("list_all_allocated_qubits", &ChipError::list_all_allocated_qubits)
		.def("assign_qubit_frequencies", &ChipError::assign_qubit_frequencies)
		.def("assign_qubit_frequencies_full", &ChipError::assign_qubit_frequencies_full)
		.def("assign_qubit_frequencies_with_ranges", &ChipError::assign_qubit_frequencies_with_ranges)
		.def("assign_qubit_frequencies_by_idx_dict", &ChipError::assign_qubit_frequencies_by_idx_dict)
		.def("assign_qubit_frequencies_by_pair_dict", &ChipError::assign_qubit_frequencies_by_pair_dict)
		.def("list_qubit_freq_ranges", &ChipError::list_qubit_freq_ranges)

		.def("initialize_all_couplers", &ChipError::initialize_all_couplers)
		.def("list_all_unallocated_couplers", &ChipError::list_all_unallocated_couplers)
		.def("list_all_allocated_couplers", &ChipError::list_all_allocated_couplers)
		.def("assign_coupler_frequencies", &ChipError::assign_coupler_frequencies)
		.def("assign_coupler_frequencies_by_idx_dict", &ChipError::assign_coupler_frequencies_by_idx_dict)
		.def("assign_coupler_frequencies_by_pair_dict", &ChipError::assign_coupler_frequencies_by_pair_dict)
		;

	py::class_<FrequencyAllocator>(m, "FrequencyAllocator")
		.def("get_instance", &FrequencyAllocator::get_instance, py::return_value_policy::reference)
		.def("get_chip", &FrequencyAllocator::get_chip, py::return_value_policy::reference)
		;

	m.def("single_err_model", &model_v1::single_err_model);
	m.def("loss_single_qubit", &model_v1::loss_single_qubit);
	m.def("loss_single_qubit_on_range", &model_v1::loss_single_qubit_on_range);
	m.def("random_loss_single_qubit", &model_v1::random_loss_single_qubit);
	m.def("random_allow_freq_loss_single_qubit", &model_v1::random_allow_freq_loss_single_qubit);

	m.def("twoq_err_model", &model_v1::twoq_err_model);
	m.def("loss_two_qubit", &model_v1::loss_two_qubit);

	py::class_<model_v1::InternalStateSingleQubit>(m, "InternalStateSingleQubit")
		//.def_readonly("T1_err", &InternalStateSingleQubit::T1_err)
		//.def_readonly("T2_err", &InternalStateSingleQubit::T2_err)
		.def_readonly("isolated_err", &model_v1::InternalStateSingleQubit::isolated_err)
		.def_readonly("XTalk_err", &model_v1::InternalStateSingleQubit::XTalk_err)
		.def_readonly("Residual_err", &model_v1::InternalStateSingleQubit::Residual_err)
		.def_readonly("allocate_fail_err", &model_v1::InternalStateSingleQubit::allocate_fail_err)
		//.def_readonly("T1_err_list", &InternalStateSingleQubit::T1_err_list)
		//.def_readonly("T2_err_list", &InternalStateSingleQubit::T2_err_list)
		.def_readonly("isolated_err_list", &model_v1::InternalStateSingleQubit::isolated_err_list)
		.def_readonly("qubit_err_list", &model_v1::InternalStateSingleQubit::qubit_err_list)
		.def_readonly("XTalk_err_list", &model_v1::InternalStateSingleQubit::XTalk_err_list)
		// .def_readonly("Residual_err_list", &InternalStateSingleQubit::Residual_err_list)
		.def_readonly("NN_residual_err_list", &model_v1::InternalStateSingleQubit::NN_residual_err_list)
		.def_readonly("NNN_residual_err_list", &model_v1::InternalStateSingleQubit::NNN_residual_err_list)
		.def_readonly("allocate_fail_err_list", &model_v1::InternalStateSingleQubit::allocate_fail_err_list)
		;

	py::class_<model_v1::InternalStateTwoQubit>(m, "InternalStateTwoQubit")
		.def_readonly("T1_err", &model_v1::InternalStateTwoQubit::T1_err)
		.def_readonly("T2_err", &model_v1::InternalStateTwoQubit::T2_err)
		.def_readonly("coupler_err_list", &model_v1::InternalStateTwoQubit::coupler_err_list)
		.def_readonly("pulse_distortion_err", &model_v1::InternalStateTwoQubit::pulse_distortion_err)
		.def_readonly("XTalk_spectator", &model_v1::InternalStateTwoQubit::XTalk_spectator)
		.def_readonly("XTalk_parallel", &model_v1::InternalStateTwoQubit::XTalk_parallel)
		.def_readonly("T1_err_list", &model_v1::InternalStateTwoQubit::T1_err_list)
		.def_readonly("T2_err_list", &model_v1::InternalStateTwoQubit::T2_err_list)
		.def_readonly("pulse_distortion_err_list", &model_v1::InternalStateTwoQubit::pulse_distortion_err_list)
		.def_readonly("XTalk_parallel_err_list", &model_v1::InternalStateTwoQubit::XTalk_parallel_err_list)
		.def_readonly("XTalk_spectator_err_list", &model_v1::InternalStateTwoQubit::XTalk_spectator_err_list)
		;
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif