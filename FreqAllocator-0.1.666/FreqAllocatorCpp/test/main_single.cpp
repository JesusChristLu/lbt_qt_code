#include <iostream>
#include "fmt/core.h"
#include "chip_graph.h"
#include "single_qubit_model.h"

int main1() 
{	
	ChipError error(8, 6);
	error.load_file(
		"../chipdata/qubit_data.json", 
		"../chipdata/xy_crosstalk_sim.json"
	);

	auto&& [loss, internal_state] = model_v1::single_err_model(error, {}, true);
	fmt::print("loss = {}\n", loss);
	fmt::print("internal_state = {}\n", internal_state.to_string());
	return 0;
}

int main2()
{
	auto basepath = std::filesystem::current_path() / ".." / ".." / ".." / "..";
	basepath = std::filesystem::canonical(basepath);
	auto qubit_datapath = basepath / "chipdata" / "qubit_data.json";
	auto xy_crosstalk_sim_datapath = basepath / "chipdata" / "xy_crosstalk_sim.json";

	std::cout << std::filesystem::current_path() << "\n";
	std::cout << basepath << "\n";
	std::cout << qubit_datapath.string().c_str() << "\n";
	std::cout << xy_crosstalk_sim_datapath.string().c_str() << "\n";

	FrequencyAllocator::get_chip().load_file(
		qubit_datapath.string().c_str(),
		xy_crosstalk_sim_datapath.string().c_str()
	);
	for (int i = 0; i < 1000; ++i)
	{
		double random_loss_single_qubit = model_v1::random_loss_single_qubit();
		fmt::print("random loss = {}\n", random_loss_single_qubit);
	}
	for (int i = 0; i < 1000; ++i)
	{
		double random_loss_single_qubit = model_v1::random_allow_freq_loss_single_qubit();
		fmt::print("random allow_freq loss = {}\n", random_loss_single_qubit);
	}
	fmt::print("{}\n", profiler::get_all_profiles_v2());
	return 0;
}

int main()
{
	std::vector<double> frequencies = { 4158.0, 4171.0, 4216.0, 4046.0, 4535.0, 4071.0, 4660.0, 4383.0, 4967.0, 4046.0, 4884.0, 4173.0, 4235.0, 4740.0, 4005.0, 4634.0, 4143.0, 4207.0, 4572.0, 4187.0, 4379.0, 4330.0, 4297.0, 4140.0, 4683.0, 4028.0, 4549.0, 4043.0, 4104.0, 4252.0, 4415.0, 4377.0, 4077.0, 4766.0, 4172.0, 4246.0, 4624.0, 4188.0, 4811.0, 4121.0, 4038.0, 4092.0, 4766.0, 3995.0, 4294.0, 4582.0, 4203.0, 4703.0, 4452.0, 4164.0, 4304.0, 4002.0, 4576.0, 4907.0, 4465.0, 4097.0, 4657.0, 4991.0, 4499.0, 4010.0, 4516.0, 4072.0, 4823.0, 4444.0, 4323.0 };

	auto basepath = std::filesystem::current_path() / ".." / ".." / ".." / "..";
	basepath = std::filesystem::canonical(basepath);
	auto qubit_datapath = basepath / "chipdata" / "qubit_data.json";
	auto xy_crosstalk_sim_datapath = basepath / "chipdata" / "xy_crosstalk_sim.json";
	auto& chip = FrequencyAllocator::get_chip();
	chip.load_file(
		qubit_datapath.string().c_str(),
		xy_crosstalk_sim_datapath.string().c_str()
	);
	chip.assign_qubit_frequencies(frequencies);

	auto&& [err, inter] = model_v1::single_err_model(chip, {}, true);

	fmt::println("Error = {}", err);
	fmt::println("Internal = {}", inter.to_string());

	return 0;
}