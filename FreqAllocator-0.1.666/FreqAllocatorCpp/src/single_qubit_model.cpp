#include "single_qubit_model.h"

namespace model_v1
{

	std::tuple<double, model_v1::InternalStateSingleQubit> single_err_model(const ChipError& chip, 
		const std::vector<double>& arb, bool record_internal)
	{
		// double mu_threshold = 0.005;
		double cost = 0.0;
		model_v1::InternalStateSingleQubit internal_state;
		if (record_internal)
			internal_state._init_lists(chip.H * chip.W);

		const double* a;
		static double arb_default[] = { 2e-4, 1e-7, 1, 0.3, 10, 1e-2, 0.5, 10 };
		if (arb.size() == 0)
			a = arb_default;
		else
			a = arb.data();		


		for (int i = 0; i < chip.H; ++i)
		{
			for (int j = 0; j < chip.W; ++j)
			{
				if (chip.unused_or_unallocated_qubit(i, j))
					// unused or unallocated, just skip 
					continue;

				int qubit_index = chip.qubit_idx(i, j);
				auto& node = chip.qubits[qubit_index];

				if (!node.frequency_in_allow_range())
				{
					cost += 1.0;
					if (record_internal)
					{
						internal_state.qubit_err_list[qubit_index] += 1.0;
						internal_state.allocate_fail_err_list[qubit_index] = 1.0;
						internal_state.allocate_fail_err += 1.0;
					}
				}

				///* Compute T1 error */
				//auto T1_err = node.model.t1_spectrum.singq_T1_err(a[0], node.t_sq, node.frequency);
				//if (T1_err < 0)
				//	ThrowRuntimeError(fmt::format("T1_err < 0 ? Qubit=({},{})", i, j));
				//cost += T1_err;
				//internal_state.T1_err += T1_err;
				//internal_state.T1_err_list[qubit_index] = T1_err;

				///* Compute T2 error */
				//auto T2_err = node.model.t2_spectrum.singq_T2_err(a[1], node.t_sq, node.frequency);
				//if (T2_err < 0)
				//	ThrowRuntimeError(fmt::format("T1_err < 0 ? Qubit=({},{})", i, j));
				//cost += T2_err;
				//internal_state.T2_err += T2_err;
				//internal_state.T2_err_list[qubit_index] = T2_err;

				/* Compute Isolated error */
				auto isolated_error = node.model.isolated_error.isolated_error(node.frequency);
				cost += isolated_error;
				if (record_internal)
				{
					internal_state.qubit_err_list[qubit_index] += isolated_error;
					internal_state.isolated_err += isolated_error;
					internal_state.isolated_err_list[qubit_index] = isolated_error;
				}

				for (int other_i = 0; other_i < chip.H; ++other_i)
				{
					for (int other_j = 0; other_j < chip.W; ++other_j)
					{
						if (chip.unused_or_unallocated_qubit(other_i, other_j))
							// unused or unallocated, just skip
							continue;
						int neighbor_index = chip.qubit_idx(other_i, other_j);
						if (neighbor_index == qubit_index)
							// itself, just skip
							continue;

						auto& neighbor_node = chip.qubits[neighbor_index];
						double detune = neighbor_node.frequency - node.frequency;
						double mu_target_neighbor = node.xy_crosstalk[neighbor_index];

						//if (mu_target_neighbor < mu_threshold)
						//	// less than mu_threshold will be skipped
						//	continue;
						auto XTalk_err = node.model.xtalk_error.singq_xtalk_err(a[2], detune, mu_target_neighbor);
						cost += XTalk_err;

						if (record_internal)
						{
							internal_state.qubit_err_list[qubit_index] += XTalk_err;
							internal_state.XTalk_err += XTalk_err;
							internal_state.XTalk_err_list[qubit_index][neighbor_index] = XTalk_err;
						}
					}
				}

				/* Compute XTalk error */
				std::vector<std::pair<int, int>> neighbors = chip.get_neighbors(i, j);
				/* Compute Residual err for nearest neighbor */
				for (auto&& [neighbor_i, neighbor_j] : neighbors)
				{
					if (chip.unused_or_unallocated_qubit(neighbor_i, neighbor_j))
						// unused or unallocated, just skip
						continue;

					int neighbor_index = chip.qubit_idx(neighbor_i, neighbor_j);
					auto& neighbor_node = chip.qubits[neighbor_index];

					auto Residual_err = node.model.residual_error.singq_residual_error(
						a[3], a[4], neighbor_node.frequency, node.frequency, neighbor_node.anharm, node.anharm);
					cost += Residual_err;
					if (record_internal) 
					{
						internal_state.qubit_err_list[qubit_index] += Residual_err;
						internal_state.Residual_err += Residual_err;
						internal_state.NN_residual_err_list[qubit_index][neighbor_index] = Residual_err;
					}
				}

				/* Compute Residual error with distance sqrt2 */
				std::vector<std::pair<int, int>> neighbors_sqrt2 = chip.get_neighbors_distance_sqrt2(i, j);

				for (auto&& [neighbor_i, neighbor_j] : neighbors_sqrt2)
				{
					if (chip.unused_or_unallocated_qubit(neighbor_i, neighbor_j))
						// unused or unallocated, just skip
						continue;

					int neighbor_index = chip.qubit_idx(neighbor_i, neighbor_j);
					auto& neighbor_node = chip.qubits[neighbor_index];

					auto Residual_err = node.model.residual_error.singq_residual_error(
						a[6], a[7], neighbor_node.frequency, node.frequency, neighbor_node.anharm, node.anharm);
					cost += Residual_err;
					if (record_internal)
					{
						internal_state.qubit_err_list[qubit_index] += Residual_err;
						internal_state.Residual_err += Residual_err;
						internal_state.NNN_residual_err_list[qubit_index][neighbor_index] = Residual_err;
					}
				}

				/* Compute Residual error with distance 2 */
				std::vector<std::pair<int, int>> neighbors_2 = chip.get_neighbors_distance_2(i, j);

				for (auto&& [neighbor_i, neighbor_j] : neighbors_2)
				{
					if (chip.unused_or_unallocated_qubit(neighbor_i, neighbor_j))
						// unused or unallocated, just skip
						continue;

					int neighbor_index = chip.qubit_idx(neighbor_i, neighbor_j);
					auto& neighbor_node = chip.qubits[neighbor_index];

					auto Residual_err = node.model.residual_error.singq_residual_error(
						a[6], a[7], neighbor_node.frequency, node.frequency, neighbor_node.anharm, node.anharm);
					cost += Residual_err;
					if (record_internal)
					{
						internal_state.qubit_err_list[qubit_index] += Residual_err;
						internal_state.Residual_err += Residual_err;
						internal_state.NNN_residual_err_list[qubit_index][neighbor_index] = Residual_err;
					}
				}
			}
		}
		return { cost, internal_state };
	}



	double loss_single_qubit(const std::vector<double>& frequencies)
	{
		profiler _("loss");
		auto& inst = FrequencyAllocator::get_instance();
		int n_dimension = inst.chip.n_available_qubits;

		if (frequencies.size() != n_dimension)
			ThrowRuntimeError(fmt::format(
				"Given frequencies does not match the full size. "
				"Given = {}, full_size = {}", frequencies.size(), n_dimension));

		inst.chip.assign_qubit_frequencies(frequencies);
		auto&& [loss, internal_state] = single_err_model(inst.chip, {}, false);

		//fmt::print("{}", internal_state.to_string());

		return loss / frequencies.size();
	}

	double loss_single_qubit_on_range(const std::vector<double>& ranges)
	{
		profiler _("loss");
		auto& chip = FrequencyAllocator::get_chip();
		int n_dimension = chip.n_available_qubits;

		if (ranges.size() != n_dimension)
			ThrowRuntimeError(fmt::format(
				"Given frequencies does not match the full size. "
				"Given = {}, full_size = {}", ranges.size(), n_dimension));

		chip.assign_qubit_frequencies_with_ranges(ranges);
		auto&& [loss, internal_state] = single_err_model(chip, {}, false);

		//fmt::print("{}", internal_state.to_string());

		return loss / ranges.size();
	}

	double random_loss_single_qubit()
	{
		profiler _("random_loss_single_qubit");
		auto& chip = FrequencyAllocator::get_chip();
		int n_dimension = chip.n_available_qubits;
		auto ranges = chip.list_qubit_freq_ranges();
		std::vector<double> freqs;
		for (auto&& [l, r] : ranges)
		{
			std::uniform_real_distribution<double> u(l, r);
			auto& rng = RandomEngine::get_instance();
			freqs.push_back(u(rng));
		}
		return loss_single_qubit(freqs);
	}

	double random_allow_freq_loss_single_qubit()
	{
		profiler _("random_allow_freq_loss_single_qubit");
		auto& chip = FrequencyAllocator::get_chip();
		auto& qubits = chip.qubits;
		
		std::vector<double> freqs;
		for (auto& node : qubits)
		{
			if (!node.used) continue;
			std::uniform_int_distribution<int> u(0, node.allow_freq.size() - 1);
			auto& rng = RandomEngine::get_instance();
			freqs.push_back(node.allow_freq[u(rng)]);
		}
		return loss_single_qubit(freqs);
	}

}