#pragma once
#include "utils.h"
#include "interpolation.h"
#include "chip_graph.h"
#include "optimizer.h"
#include "single_qubit_model.h"


namespace model_v1
{
	struct InternalStateTwoQubit
	{
		double T1_err = 0.0;
		double T2_err = 0.0;
		double pulse_distortion_err = 0.0;
		double XTalk_spectator = 0.0;
		double XTalk_parallel = 0.0;
		double inner_leakage_err = 0.0;
		
		std::vector<double> coupler_err_list;
		std::vector<double> T1_err_list;
		std::vector<double> T2_err_list;
		std::vector<double> pulse_distortion_err_list;
		std::vector<double> XTalk_spectator_err_list;
		std::vector<double> XTalk_parallel_err_list;
		std::vector<double> inner_leakage_err_list;

		void _init_list(int size)
		{
			coupler_err_list.resize(size, 0);
			T1_err_list.resize(size, 0);
			T2_err_list.resize(size, 0);
			pulse_distortion_err_list.resize(size, 0);
			XTalk_parallel_err_list.resize(size, 0);
			XTalk_spectator_err_list.resize(size, 0);
			inner_leakage_err_list.resize(size, 0);
		}

		std::string to_string() const
		{
			return fmt::format(
				"T1_err = {}\n"
				"T2_err = {}\n"
				"pulse_distortion_err = {}\n"
				"XTalk_spectator = {}\n"
				"XTalk_parallel = {}\n"
				"inner_leakage_err = {}\n",
				T1_err, T2_err,
				pulse_distortion_err,
				XTalk_spectator,
				XTalk_parallel, inner_leakage_err
			);

		}
	};

	std::tuple<double, InternalStateTwoQubit> twoq_err_model(const ChipError& chip, const std::vector<double>& a, bool record_internal);

	double loss_two_qubit(const std::vector<double>& frequencies);
}