#pragma once
#include "utils.h"
#include "interpolation.h"
#include "chip_graph.h"
#include "optimizer.h"

// ***************************************************************
// To decouple the model from a static data struct, any new model
// will not be set as a member function for ChipError.
// Instead, they will be a function here. A typical case should be
// double Loss = SomeLossFunc(ChipError& chip);
// 
// Note 1: all attributes for ChipError is exposed to all, use
//   it safely without changing any unnecessary data.
//
// Note 2: To make a allocation, just assign every ChipQubit.frequency
//    Before they are allocated, they are set to -1.0, supporting
//    the computing loss of partial allocation.
//  
// ****************************************************************

namespace model_v1
{
	struct InternalStateSingleQubit
	{
		//double T1_err = 0.0;
		//double T2_err = 0.0;
		double isolated_err = 0.0;
		double XTalk_err = 0.0;
		double Residual_err = 0.0;
		double allocate_fail_err = 0.0;

		//std::vector<double> T1_err_list;
		//std::vector<double> T2_err_list;
		std::vector<double> qubit_err_list;
		std::vector<double> isolated_err_list;
		std::vector<std::map<int, double>> XTalk_err_list;
		std::vector<std::map<int, double>> NN_residual_err_list;
		std::vector<std::map<int, double>> NNN_residual_err_list;
		std::vector<double> allocate_fail_err_list;

		inline void _init_lists(int size)
		{
			//T1_err_list.resize(size);
			//T2_err_list.resize(size);
			isolated_err_list.resize(size);
			XTalk_err_list.resize(size);
			NN_residual_err_list.resize(size);
			NNN_residual_err_list.resize(size);
			// Residual_err_list.resize(size);
			allocate_fail_err_list.resize(size);
			qubit_err_list.resize(size);
		}

		inline std::string to_string() const
		{
			return fmt::format(
				/*"T1_err = {}\n"
				"T2_err = {}\n"*/
				"isolated_err = {}\n"
				"XTalk_err = {}\n"
				"Residual_err = {}\n"
				"allocate_fail_err = {}\n",
				/*T1_err, T2_err,*/
				isolated_err,
				XTalk_err, Residual_err, allocate_fail_err
			);
		}
	};

	std::tuple<double, model_v1::InternalStateSingleQubit> single_err_model(const ChipError& chip, const std::vector<double>& arb, bool record_internal);

	double loss_single_qubit(const std::vector<double>& frequencies);
	double loss_single_qubit_on_range(const std::vector<double>& range);
	double random_loss_single_qubit();
	double random_allow_freq_loss_single_qubit();

}
