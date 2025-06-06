#include "utils.h"
#include "formula.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

inline bool get_allow_freq(const rapidjson::Value& json, std::vector<double>& allow_freq)
{
	if (!json.HasMember("allow_freq"))
	{
		return false;
	}
	allow_freq.clear();
	for (auto& v : json["allow_freq"].GetArray())
	{
		allow_freq.push_back(v.GetDouble());
	}
	return true;
}

inline bool get_isolated_error(const rapidjson::Value& json, std::vector<double>& isolated_error)
{
	if (!json.HasMember("isolated_error"))
	{
		return false;
	}
	isolated_error.clear();
	for (auto& v : json["isolated_error"].GetArray())
	{
		isolated_error.push_back(v.GetDouble());
	}
	return true;
}

inline bool get_bad_freq_range(const rapidjson::Value& json, std::vector<std::pair<double, double>>& bad_freq_range)
{
	if (!json.HasMember("bad_freq_range"))
	{
		return false;
	}
	bad_freq_range.clear();
	if (json["bad_freq_range"].Empty())
	{
		return true;
	}
	for (auto& freq_range : json["bad_freq_range"].GetArray())
	{
		double left;
		double right;
		
		if (freq_range.GetArray().Size() != 2)
			return false;

		left = freq_range.GetArray()[0].GetDouble();
		right = freq_range.GetArray()[0].GetDouble();

		bad_freq_range.emplace_back(left, right);
	}
	return true;
}

inline bool get_ac_spectrum(const rapidjson::Value& json, AcSpectrumParam& ac_spectrum)
{
	if (!json.HasMember("ac_spectrum"))
	{
		return false;
	}
	auto& ac_spectrum_json = json["ac_spectrum"].GetArray();

	if (ac_spectrum_json.Size() == 5)
	{
		ac_spectrum.fq_max = ac_spectrum_json[0].GetDouble();
		ac_spectrum.detune = ac_spectrum_json[1].GetDouble();
		ac_spectrum.M = ac_spectrum_json[2].GetDouble();
		ac_spectrum.offset = ac_spectrum_json[3].GetDouble();
		ac_spectrum.d = ac_spectrum_json[4].GetDouble();

		ac_spectrum.freq_max = ac_spectrum.fq_max;
		ac_spectrum.freq_min = amp2freq_formula(M_PI/2, ac_spectrum, true);
		return true;
	}
	else if (ac_spectrum_json.Size() == 11)
	{
		ac_spectrum.fq_max = ac_spectrum_json[0].GetDouble();
		ac_spectrum.detune = ac_spectrum_json[1].GetDouble();
		ac_spectrum.M = ac_spectrum_json[2].GetDouble();
		ac_spectrum.offset = ac_spectrum_json[3].GetDouble();
		ac_spectrum.d = ac_spectrum_json[4].GetDouble();
		ac_spectrum.w = ac_spectrum_json[5].GetDouble();
		ac_spectrum.g = ac_spectrum_json[6].GetDouble();

		ac_spectrum.freq_max = ac_spectrum_json[10].GetDouble();
		ac_spectrum.freq_min = amp2freq_formula(M_PI / 2, ac_spectrum, true);
		return true;
	}

	return false;

}

inline bool get_t1_spectrum(const rapidjson::Value& json, std::vector<double>& t1_freq, std::vector<double> &t1_t1)
{
	if (!json.HasMember("t1_spectrum"))
	{
		return false;
	}
	t1_freq.clear();
	t1_t1.clear();

	if (!json["t1_spectrum"].HasMember("freq"))
		return false;

	const auto& freq = json["t1_spectrum"]["freq"];

	if (!json["t1_spectrum"].HasMember("t1"))
		return false;

	const auto& t1 = json["t1_spectrum"]["t1"];

	for (auto& v : freq.GetArray())
	{
		t1_freq.push_back(v.GetDouble());
	}

	for (auto& v : t1.GetArray())
	{
		t1_t1.push_back(v.GetDouble());
	}
	return true;
}

inline bool get_xy_crosstalk_coef(const rapidjson::Value& json, std::vector<double>& xy_crosstalk)
{
	if (!json.HasMember("xy_crosstalk_coef"))
	{
		return false;
	}
	xy_crosstalk.clear();
	xy_crosstalk.resize(72, 0);
	
	const rapidjson::Value& xy_crosstalk_json = json["xy_crosstalk_coef"];

	for (int i = 0; i < 72; ++i)
	{
		std::string qname = fmt::format("q{}", i + 1);
		if (xy_crosstalk_json.HasMember(qname.c_str()))
		{
			xy_crosstalk[i] = xy_crosstalk_json[qname.c_str()].GetDouble();
		}
	}

	return true;
}

inline bool get_anharm(const rapidjson::Value& json, double &anharm)
{
	if (!json.HasMember("anharm"))
	{
		return false;
	}
	
	anharm = std::round(json["anharm"].GetDouble());
	return true;
}

inline bool _load_qubit_info_from_json(const rapidjson::Document& d, int qubit_idx,
	std::vector<double>& allow_freq,
	std::vector<double>& isolated_error,
	AcSpectrumParam& ac_spectrum,
	std::vector<double>& t1_freq,
	std::vector<double>& t1_t1,
	std::vector<double>& xy_crosstalk,
	double& anharm
	)
{
	std::string qubit = fmt::format("q{}", qubit_idx);
	if (!d.HasMember(qubit.c_str()))
	{
		return false;
	}
	const rapidjson::Value& qubitjson = d[qubit.c_str()];
	
	if (!get_allow_freq(qubitjson, allow_freq))
		ThrowRuntimeError(fmt::format("allow_freq of {} has error in handling json.", qubit));

	if (!get_isolated_error(qubitjson, isolated_error))
		ThrowRuntimeError(fmt::format("isolated_error of {} has error in handling json.", qubit));

	if (!get_ac_spectrum(qubitjson, ac_spectrum))
		ThrowRuntimeError(fmt::format("ac_spectrum of {} has error in handling json.", qubit));

	if (!get_t1_spectrum(qubitjson, t1_freq, t1_t1))
		ThrowRuntimeError(fmt::format("t1_spectrum of {} has error in handling json.", qubit));

	if (!get_xy_crosstalk_coef(qubitjson, xy_crosstalk))
		ThrowRuntimeError(fmt::format("t1_spectrum of {} has error in handling json.", qubit));

	if (!get_anharm(qubitjson, anharm))
		ThrowRuntimeError(fmt::format("anharm of {} has error in handling json.", qubit));

	return true;
}

//
//inline void load_xy_crosstalk(const rapidjson::Document& d, std::vector<std::vector<double>> &xy_crosstalk_matrix)
//{
//	// Note: assume the xy_crosstalk (from qubit_data.json) is always a 72x72 matrix.
//	constexpr int N_ROW = 72;
//	constexpr int N_COLUMN = 72;
//
//	if (!d.HasMember("xy_crosstalk_coef"))
//	{
//		ThrowRuntimeError("Cannot find xy_crosstalk in json.");
//	}
//	if (!d["xy_crosstalk_coef"].HasMember("xy_crosstalk"))
//	{
//		ThrowRuntimeError("Cannot find xy_crosstalk in json.");
//	}
//
//	const rapidjson::Value& xy_crosstalk_json = d["xy_crosstalk"]["xy_crosstalk"];
//	const auto& arrs = xy_crosstalk_json.GetArray();
//	if (arrs.Size() != N_COLUMN)
//	{
//		ThrowRuntimeError("XY crosstalk is not 72x72.");
//	}
//	xy_crosstalk_matrix.clear();
//	xy_crosstalk_matrix.reserve(72);
//	for (auto& arr : arrs)
//	{
//		if (arr.Size() != N_ROW)
//		{
//			ThrowRuntimeError("XY crosstalk is not 72x72.");
//		}
//		std::vector<double> xy_crosstalk_row;
//		xy_crosstalk_row.reserve(72);
//		for (auto& v : arr.GetArray())
//		{
//			xy_crosstalk_row.push_back(v.GetDouble());
//		}
//		xy_crosstalk_matrix.emplace_back(std::move(xy_crosstalk_row));
//	}
//}

inline bool get_alpha_list(const rapidjson::Document& json, std::vector<double>& alpha_list)
{
	if (!json.HasMember("alpha_list"))
	{
		return false;
	}
	alpha_list.clear();
	for (auto& v : json["alpha_list"].GetArray())
	{
		alpha_list.push_back(v.GetDouble());
	}
	return true;
}


inline bool get_mu_list(const rapidjson::Document& json, std::vector<double>& mu_list)
{
	if (!json.HasMember("mu_list"))
	{
		return false;
	}
	mu_list.clear();
	for (auto& v : json["mu_list"].GetArray())
	{
		mu_list.push_back(v.GetDouble());
	}
	return true;
}

inline bool get_detune_list(const rapidjson::Document& json, std::vector<double>& detune_list)
{
	if (!json.HasMember("detune_list"))
	{
		return false;
	}
	detune_list.clear();
	for (auto& v : json["detune_list"].GetArray())
	{
		detune_list.push_back(v.GetDouble());
	}
	return true;
}

inline bool get_error_arr(const rapidjson::Document& json, std::vector<std::vector<std::vector<double>>>& error_arr)
{
	if (!json.HasMember("error_arr"))
	{
		return false;
	}
	error_arr.clear();
	for (auto& arr_json : json["error_arr"].GetArray())
	{
		std::vector<std::vector<double>> arr;
		for (auto& vec_json : arr_json.GetArray())
		{
			std::vector<double> vec;
			vec.reserve(vec_json.Size());
			for (auto& value : vec_json.GetArray())
			{
				vec.push_back(value.GetDouble());
			}
			arr.push_back(vec);
		}
		error_arr.push_back(arr);
	}
	
	return true;
}
