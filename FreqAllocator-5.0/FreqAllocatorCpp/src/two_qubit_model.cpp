#include "two_qubit_model.h"


static double cross_coupler_xtalk(int q0, int q1, int qh, int ql, 
	int qh_neighbor, int ql_neighbor,
	double pulseql, double pulseqh,
	const ChipError& chip, const ChipCoupler& coupler, const ChipCoupler& coupler_neighbor,
	const double* a)
{
	// compute the xtalk error of q0 and q1 (q0 in the computed coupler and q1 in another)

	// when two qubits are not connected, return 0
	if (chip.get_distance(q0, q1) != 1) return 0.0;

	double cost = 0;

	auto& qubit0 = chip.qubits[q0];
	auto& qubit1 = chip.qubits[q1];

	double fwork, fidle, fwork_neighbor, fidle_neighbor;

	// decide whether q0 is ql or qh
	double pulse;
	if (q0 == ql) {
		pulse = pulseql;
	}
	else {
		pulse = pulseqh;
	}

	double pulse_neighbor;
	if (q1 == ql_neighbor)
	{
		pulse_neighbor = coupler_neighbor.frequency;
	}
	else
	{
		pulse_neighbor = coupler_neighbor.frequency - qubit1.anharm;
	}	


	cost = coupler.model.xtalk_error.twoq_xtalk_err(
		pulse, pulse_neighbor,
		a, 
		qubit0.anharm, qubit1.anharm
	);

	return cost;
}

static double self_coupler_xtalk(int qubit_idx, int qh, int ql, double pulseqh, double pulseql, const ChipError& chip, const ChipCoupler& coupler, const double* a)
{
	auto& qubit = chip.qubits[qubit_idx];
	double pulse;
	if (qubit_idx == ql)
	{
		pulse = pulseql;
	}
	else if (qubit_idx == qh)
	{
		pulse = pulseqh;
	}
	else
		throw std::runtime_error("Unmatch qubit_idx in computing self_coupler_xtalk.");

	double cost = 0;
	auto&& [i, j] = chip.qubit_idx_to_pos(qubit_idx);
	// get neighbors of qubit_idx
	auto neighbors = chip.get_neighbors(i, j);

	for (const auto& neighbor : neighbors)
	{
		int neighbor_idx = chip.qubit_idx(neighbor.first, neighbor.second);
		
		// ignore the node within this coupler
		if (neighbor_idx == qh || neighbor_idx == ql) 
			continue;

		// ignore the node not allocated or unused
		if (chip.unused_or_unallocated_qubit(neighbor_idx))
			continue;

		auto& neighbor_node = chip.qubits[neighbor_idx];

		cost += coupler.model.xtalk_error.twoq_xtalk_err(
			pulse,
			neighbor_node.frequency,
			a, 
			qubit.anharm, neighbor_node.anharm
		);
	}
	return cost;
}

std::tuple<double, model_v1::InternalStateTwoQubit> model_v1::twoq_err_model(
	const ChipError& chip, const std::vector<double>& axeb, bool record_internal)
{
	model_v1::InternalStateTwoQubit internal_state;
	if (record_internal)
		internal_state._init_list(chip.couplers.size());

	double cost = 0;

	const double* a;
	static double axeb_default[] = { 4e-4, 1e-7, 1e-2, 1e-5, 1e-2, 1, 10, 0.7, 10 };
	if (axeb.size() == 0)
		a = axeb_default;
	else
		a = axeb.data();
	for (auto& coupler : chip.couplers)
	{
		int q1 = coupler.qubit1;
		const ChipQubit& qubit1 = chip.qubits[q1];
		int q2 = coupler.qubit2;
		const ChipQubit& qubit2 = chip.qubits[q2];

		int coupler_id = coupler.coupler_id;

		// if this is an unused allocator, only any qubit within the coupler is not used.
		if (chip.unused_or_unallocated_coupler(coupler_id))
			continue;
		const ChipCoupler& coupler = chip.couplers[coupler_id];

		int qh, ql;
		if (qubit1.frequency > qubit2.frequency)
		{
			qh = coupler.qubit1;
			ql = coupler.qubit2;
		}
		else
		{
			qh = coupler.qubit2;
			ql = coupler.qubit1;
		}

		
		if (chip.qubits[qh].freq_max + chip.qubits[qh].anharm < chip.qubits[ql].freq_min)
			std::swap(ql, qh);

		const ChipQubit& qubit_high = chip.qubits[qh];
		const ChipQubit& qubit_low = chip.qubits[ql];

		double fwork = coupler.frequency;
		double pulseql = fwork;
		double pulseqh = fwork - chip.qubits[qh].anharm;

		double t1qh, t1ql, t2qh, t2ql;
		if (q1 == qh)
		{
			t1qh = coupler.model.t1_error_q1.twoq_T1_err(pulseqh, a[0], coupler.t_twoq);
			t1ql = coupler.model.t1_error_q2.twoq_T1_err(pulseql, a[0], coupler.t_twoq);
			t2qh = coupler.model.t2_error_q1.twoq_T2_err(pulseqh, a[1], coupler.t_twoq, qubit_high.ac_spectrum);
			t2ql = coupler.model.t2_error_q2.twoq_T2_err(pulseql, a[1], coupler.t_twoq, qubit_low.ac_spectrum);
		}
		else
		{
			t1qh = coupler.model.t1_error_q2.twoq_T1_err(pulseqh, a[0], coupler.t_twoq);
			t1ql = coupler.model.t1_error_q1.twoq_T1_err(pulseql, a[0], coupler.t_twoq);
			t2qh = coupler.model.t2_error_q2.twoq_T2_err(pulseqh, a[1], coupler.t_twoq, qubit_high.ac_spectrum);
			t2ql = coupler.model.t2_error_q1.twoq_T2_err(pulseql, a[1], coupler.t_twoq, qubit_low.ac_spectrum);
		}
		
		
		double distortion = coupler.model.distort_error.twoq_pulse_distort_err(
			{ pulseqh, qubit_high.frequency },
			{ pulseql, qubit_low.frequency },
			a[2], 
			qubit_high.ac_spectrum, 
			qubit_low.ac_spectrum);

		double inner_leakage = coupler.model.inner_leakage_error.inner_leakage(
			{ pulseqh, qubit_high.frequency },
			{ pulseql, qubit_low.frequency },
			a[3], a[4]
		);

		cost += t1ql;
		cost += t1qh;
		cost += t2ql;
		cost += t2qh;
		cost += distortion;
		cost += inner_leakage;
		if (record_internal) {
			internal_state.T1_err += t1ql;
			internal_state.T1_err += t1qh;
			internal_state.T2_err += t2ql;
			internal_state.T2_err += t2qh;
			internal_state.pulse_distortion_err += distortion;
			internal_state.inner_leakage_err += inner_leakage;

			internal_state.T1_err_list[coupler_id] += t1ql;
			internal_state.T1_err_list[coupler_id] += t1qh;
			internal_state.T2_err_list[coupler_id] += t2ql;
			internal_state.T2_err_list[coupler_id] += t2qh;
			internal_state.pulse_distortion_err_list[coupler_id] += distortion;
			internal_state.inner_leakage_err_list[coupler_id] += inner_leakage;

			internal_state.coupler_err_list[coupler_id] += t1ql;
			internal_state.coupler_err_list[coupler_id] += t1qh;
			internal_state.coupler_err_list[coupler_id] += t2ql;
			internal_state.coupler_err_list[coupler_id] += t2qh;
			internal_state.coupler_err_list[coupler_id] += distortion;
			internal_state.coupler_err_list[coupler_id] += inner_leakage;
		}

		// compute xtalk from self coupler		
		// a+5 represents a[5:]
		double xtalk_from_q1 = self_coupler_xtalk(q1, qh, ql, pulseqh, pulseql, chip, coupler, a + 5);
		double xtalk_from_q2 = self_coupler_xtalk(q2, qh, ql, pulseqh, pulseql, chip, coupler, a + 5);

		cost += xtalk_from_q1;
		cost += xtalk_from_q2;
		
		if (record_internal)
		{
			internal_state.XTalk_spectator += xtalk_from_q1;
			internal_state.XTalk_spectator += xtalk_from_q2;
			internal_state.XTalk_spectator_err_list[coupler_id] += xtalk_from_q1;
			internal_state.XTalk_spectator_err_list[coupler_id] += xtalk_from_q2;
			internal_state.coupler_err_list[coupler_id] += xtalk_from_q1;
			internal_state.coupler_err_list[coupler_id] += xtalk_from_q2;
		}

		// from coupler's neighbor couplers
		auto &neighbor_couplers = coupler.neighbor_couplers;
		for (auto& neighbor_coupler_idx : neighbor_couplers)
		{
			auto& neighbor_coupler = chip.couplers[neighbor_coupler_idx];

			// ignore the unused coupler
			if (chip.unused_or_unallocated_coupler(neighbor_coupler_idx))
				continue;

			int q3 = neighbor_coupler.qubit1;
			int q4 = neighbor_coupler.qubit2;

			// decide whether the neighbor coupler 
			int ql_neighbor, qh_neighbor;
			auto& qubit0_neighbor = chip.qubits[neighbor_coupler.qubit1];
			auto& qubit1_neighbor = chip.qubits[neighbor_coupler.qubit2];
			if (qubit0_neighbor.frequency < qubit1_neighbor.frequency)
			{
				ql_neighbor = neighbor_coupler.qubit1;
				qh_neighbor = neighbor_coupler.qubit2;
			}
			else
			{
				qh_neighbor = neighbor_coupler.qubit1;
				ql_neighbor = neighbor_coupler.qubit2;
			}

			/* Note for cross_coupler_xtalk */

			/* qh, ql  are the index of the h/l qubit in "this" coupler */
			/* qh_neighbor, ql_neighbor  are the index of the h/l qubit in "neighbor" coupler */
			/* pulse_qh and pulse_ql are the pulse parameters for h/l qubit */
			/* other are global parameters */
			// a+5 represents a[5:]
			double xtalk_from_q1q3 = cross_coupler_xtalk(q1, q3, qh, ql, qh_neighbor, ql_neighbor, pulseql, pulseqh, chip, coupler, neighbor_coupler, a + 5);
			double xtalk_from_q2q3 = cross_coupler_xtalk(q2, q3, qh, ql, qh_neighbor, ql_neighbor, pulseql, pulseqh, chip, coupler, neighbor_coupler, a + 5);
			double xtalk_from_q1q4 = cross_coupler_xtalk(q1, q4, qh, ql, qh_neighbor, ql_neighbor, pulseql, pulseqh, chip, coupler, neighbor_coupler, a + 5);
			double xtalk_from_q2q4 = cross_coupler_xtalk(q2, q4, qh, ql, qh_neighbor, ql_neighbor, pulseql, pulseqh, chip, coupler, neighbor_coupler, a + 5);

			cost += xtalk_from_q1q3;
			cost += xtalk_from_q2q3;
			cost += xtalk_from_q1q4;
			cost += xtalk_from_q2q4;
			if (record_internal)
			{
				internal_state.XTalk_parallel += xtalk_from_q1q3;
				internal_state.XTalk_parallel += xtalk_from_q2q3;
				internal_state.XTalk_parallel += xtalk_from_q1q4;
				internal_state.XTalk_parallel += xtalk_from_q2q4;
				internal_state.XTalk_parallel_err_list[coupler_id] += xtalk_from_q1q3;
				internal_state.XTalk_parallel_err_list[coupler_id] += xtalk_from_q2q3;
				internal_state.XTalk_parallel_err_list[coupler_id] += xtalk_from_q1q4;
				internal_state.XTalk_parallel_err_list[coupler_id] += xtalk_from_q2q4;
				internal_state.coupler_err_list[coupler_id] += xtalk_from_q1q3;
				internal_state.coupler_err_list[coupler_id] += xtalk_from_q2q3;
				internal_state.coupler_err_list[coupler_id] += xtalk_from_q1q4;
				internal_state.coupler_err_list[coupler_id] += xtalk_from_q2q4;
			}
		}

	}
	return { cost, internal_state };
}

double model_v1::loss_two_qubit(const std::vector<double>& frequencies)
{
	auto& chip = FrequencyAllocator::get_chip();
	chip.assign_coupler_frequencies(frequencies);
	
	auto&& [loss, _] = twoq_err_model(chip, {}, false);
	return loss / frequencies.size();
}
