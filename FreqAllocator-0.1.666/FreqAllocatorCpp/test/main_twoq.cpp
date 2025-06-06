#include <iostream>
#include "fmt/core.h"
#include "chip_graph.h"
#include "single_qubit_model.h"
#include "two_qubit_model.h"

std::vector<double> load_preset_qubit_frequencies()
{	
	std::string qubit_freq_real_str = 
R"({
		"q1":  4175,
		"q3" : 4660,
		"q4" : 4107,
		"q5" : 4181,
		"q7" : 4094,
		"q8" : 4618,
		"q9" : 4146,
		"q10" : 4978,
		"q11" : 4260,
		"q12" : 4799,
		"q13" : 4658,
		"q14" : 4000,
		"q15" : 4514,
		"q16" : 4075,
		"q17" : 4624,
		"q18" : 4147,
		"q19" : 4302,
		"q20" : 4098,
		"q21" : 4187.823,
		"q23" : 4215,
		"q24" : 4743,
		"q25" : 4491,
		"q26" : 4122,
		"q27" : 4653,
		"q28" : 4106,
		"q29" : 4568,
		"q30" : 4036,
		"q31" : 4159,
		"q32" : 4570,
		"q33" : 4007,
		"q34" : 4620,
		"q35" : 4072,
		"q36" : 4454,
		"q37" : 4622,
		"q38" : 4045,
		"q39" : 4498,
		"q40" : 4178,
		"q41" : 4763,
		"q42" : 4083,
		"q43" : 4244,
		"q45" : 4333,
		"q46" : 4804,
		"q47" : 4001.0,
		"q48" : 4391,
		"q49" : 4178,
		"q50" : 4055,
		"q51" : 4699,
		"q53" : 4464,
		"q55" : 4087,
		"q56" : 4578,
		"q57" : 4419,
		"q58" : 4394,
		"q60" : 4970,
		"q61" : 4615,
		"q62" : 4003,
		"q63" : 4532,
		"q65" : 4673.0,
		"q66" : 4078,
		"q67" : 4204,
		"q68" : 4663,
		"q69" : 4085,
		"q71" : 4227,
		"q72" : 4642
})";

	rapidjson::Document d;
	d.Parse(qubit_freq_real_str.c_str());
	std::vector<double> frequencies(72, -1);
	for (int i = 0; i < 72; ++i)
	{
		std::string q_stdstr = fmt::format("q{:d}", i + 1);
		const char* qstr = q_stdstr.c_str();
		if (d.HasMember(qstr))
		{
			double v = d[qstr].GetDouble();
			frequencies[i] = v;
		}
	}

	return frequencies;
}

std::map<int, std::map<std::string, double>> load_preset_coupler_frequencies()
{
	std::map<int, std::map<std::string, double>> ret;
	std::string frequencies_str = R"(


{"8": {"frequency": 4372.0, "error_all": 0.010598179862104193}, "16": {"frequency": 4271.0, "error_all": 0.00968697208263811}, "18": {"frequency": 4255.0, "error_all": 0.007387164598044766}, "20": {"frequency": 4300.0, "error_all": 0.009685177854335614}, "28": {"frequency": 4007.0, "error_all": 0.011459151688380752}, "32": {"frequency": 4482.0, "error_all": 0.008766905846294273}, "38": {"frequency": 4377.0, "error_all": 0.009427928535602462}, "40": {"frequency": 4187.823, "error_all": 0.010032511302601837}, "42": {"frequency": 4200.0, "error_all": 0.007705615476346146}, "50": {"frequency": 4297.0, "error_all": 0.009992295655148435}, "52": {"frequency": 4278.0, "error_all": 0.009433708901180093}, "54": {"frequency": 4036.0, "error_all": 0.007606772731979113}, "60": {"frequency": 4141.0, "error_all": 0.008560825346661607}, "62": {"frequency": 3983.0, "error_all": 0.009419990075913614}, "64": {"frequency": 4135.0, "error_all": 0.0132994254586579}, "74": {"frequency": 4426.0, "error_all": 0.012459177370308685}, "76": {"frequency": 4089.0, "error_all": 0.008893748875889802}, "82": {"frequency": 4283.0, "error_all": 0.007466455199871765}, "84": {"frequency": 4333.0, "error_all": 0.009696922425110666}, "86": {"frequency": 4051.0, "error_all": 0.007335465505632589}, "94": {"frequency": 4365.0, "error_all": 0.010558608626838635}, "104": {"frequency": 3900.0, "error_all": 0.00887724692267386}, "106": {"frequency": 4428.0, "error_all": 0.00880637759165074}, "116": {"frequency": 4011.0, "error_all": 0.009111317715395731}, "120": {"frequency": 4079.0, "error_all": 0.005166000555060529}}


)";

	rapidjson::Document d;
	d.Parse(frequencies_str.c_str());
	
	for (auto& object : d.GetObject())
	{
		auto coupler_id_str = object.name.GetString();
		int coupler_id = atoi(coupler_id_str);
		double frequency = object.value["frequency"].GetDouble();
		double error = object.value["error_all"].GetDouble();
		
		ret[coupler_id] = { { "frequency", frequency }, {"error_all", error} };		
	}
	return ret;
}

std::vector<double> get_frequencies_list(const std::map<int, std::map<std::string, double>>& freq_dict)
{
	std::vector<double> frequencies(126, -1);

	for (auto coupler : freq_dict)
	{
		frequencies[coupler.first] = coupler.second["frequency"];
	}

	return frequencies;
}

int main()
{
	auto basepath = std::filesystem::current_path() / ".." / ".." / ".." / "..";
	basepath = std::filesystem::canonical(basepath);
	auto qubit_datapath = basepath / "chipdata" / "qubit_data.json";
	auto xy_crosstalk_sim_datapath = basepath / "chipdata" / "xy_crosstalk_sim.json";
	auto& chip = FrequencyAllocator::get_chip();
	chip.load_file(
		qubit_datapath.string().c_str(),
		xy_crosstalk_sim_datapath.string().c_str()
	);

	auto frequencies = load_preset_qubit_frequencies();
	auto coupler_frequencies_dict = load_preset_coupler_frequencies();

	chip.assign_qubit_frequencies_full(frequencies);
	auto&& [err, inter] = model_v1::single_err_model(chip, {}, true);

	fmt::println("{}", frequencies);
	fmt::println("Single Error =	 {}", err);
	fmt::println("Single Internal\n {}", inter.to_string());

	fmt::println("{}", coupler_frequencies_dict);
	auto coupler_frequencies = get_frequencies_list(coupler_frequencies_dict);

	coupler_frequencies = std::vector<double>
	{ 

		-1, -1, -1, -1, -1, 3919.0, -1, 3964.0, -1, 4173.0, -1, -1, -1, -1, -1, -1, -1, 4006.0, -1, 4345.0, -1, 4123.0, -1, -1, -1, -1, -1, 4420.0, -1, 4187.823, -1, 4378.0, -1, -1, -1, -1, -1, -1, -1, 4125.0, -1, -1, -1, 4036.0, -1, -1, -1, -1, -1, 3992.0, -1, 4027.0, -1, 4309.0, -1, -1, -1, -1, -1, -1, -1, 4316.0, -1, 4426.0, -1, 4227.0, -1, -1, -1, -1, -1, 4280.0, -1, 4376.0, -1, 4346.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3900.0, -1, 3948.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4009.0, -1, -1, -1, 4574.0, -1, -1, -1, -1, -1, 4312.0, -1, 4298.0, -1, 4478.0, -1, -1, -1, -1, -1, -1

	};
	chip.assign_coupler_frequencies(coupler_frequencies);
	auto&& [err2q, inter2q] = model_v1::twoq_err_model(chip, {}, true);
	fmt::println("Two Error = {}", err2q);
	fmt::println("Two Internal\n{}", inter2q.to_string());
	fmt::println("All error\n{}", inter2q.coupler_err_list);
	fmt::println("Pulse Distortion\n{}", inter2q.pulse_distortion_err_list);
	fmt::println("XTalk Spectator\n{}", inter2q.XTalk_spectator_err_list);
	fmt::println("XTalk Parallel\n{}", inter2q.XTalk_parallel_err_list);
	/*auto loss = model_v1::loss_two_qubit(coupler_frequencies);
	// fmt::println("Loss = {}", loss);*/

	return 0;
}

int main1()
{
	auto basepath = std::filesystem::current_path() / ".." / ".." / ".." / "..";
	basepath = std::filesystem::canonical(basepath);
	auto qubit_datapath = basepath / "chipdata" / "qubit_data.json";
	auto xy_crosstalk_sim_datapath = basepath / "chipdata" / "xy_crosstalk_sim.json";
	auto& chip = FrequencyAllocator::get_chip();
	chip.load_file(
		qubit_datapath.string().c_str(),
		xy_crosstalk_sim_datapath.string().c_str()
	);

	for (auto& coupler : chip.couplers)
	{
		auto&& [q1, q2] = chip.from_coupler_idx_to_qubits(coupler.coupler_id);
		fmt::println("{}: {}, {} neighbor : {}", coupler.coupler_id, q1, q2, coupler.neighbor_couplers);
	}
	return 0;

}