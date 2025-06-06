#include "chip_graph.h"
#include "Profiler.h"

void T1Spectrum::_init(const std::vector<double>& f_list_, const std::vector<double>& t1_list_)
{
    f_list = f_list_;
    t1_list = t1_list_;
    // func_interp = CubicSpline(f_list_, t1_list_);
    func_interp = Interp1d(f_list_, t1_list_);
}

T1Spectrum::T1Spectrum(const std::vector<double>& f_list_, const std::vector<double>& t1_list_)
    : f_list(f_list), t1_list(t1_list), func_interp(f_list, t1_list)
{
}

double T1Spectrum::singq_T1_err(double a, double tq, double freq) const
{
    profiler _("singq_T1_err");
    try {
        double error = a * tq / func_interp(freq);
        if (error < 0)
            return 5e-4;
        return error;
    }
    catch (std::out_of_range& e)
    {
        return 5e-4;
    }
}

void T2Spectrum::_init(const std::vector<double>& f_list_, const std::vector<double>& t2_list_)
{
    if (f_list.size() != 0)
    {
        f_list = f_list_;
        t2_list = t2_list_;
        use_dummy_t2_spectrum = false;
        func_interp = CubicSpline(f_list, t2_list);
    }
    else
    {
        use_dummy_t2_spectrum = true;
    }
}

T2Spectrum::T2Spectrum(const std::vector<double>& f_list_, 
    const std::vector<double>& t2_list_)
{
    _init(f_list_, t2_list_);
}

double T2Spectrum::singq_T2_err(double a, double tq, double f, const AcSpectrumParam& params) const
{
    profiler _("singq_T2_err");
    if (!use_dummy_t2_spectrum)
    {
        return a * tq * func_interp(f);
    }
    else
    {
        double df_dphi = 1.0 / (
            std::fabs(freq2amp_formula(f, params, true) - freq2amp_formula(f - 0.01, params, true)) / 0.01
            );
        double error = a * tq * df_dphi;
        if (std::isnan(error))
            return 5.0e-4;
        else
            return error;
    }
}

void XTalkError1q::_init(const std::vector<double>& alpha_list_,
    const std::vector<double>& mu_list_,
    const std::vector<double>& detune_list_,
    const std::vector<std::vector<std::vector<double>>> error_arr_,
    double anharm_) 
{
    alpha_list = alpha_list_;
    mu_list = mu_list_;
    detune_list = detune_list_;
    error_arr = error_arr_;

    anharm = anharm_;
    // auto iter = std::lower_bound(alpha_list.begin(), alpha_list.end(), anharm, [](double a, double b) {return a > b; });
    auto iter = std::find(alpha_list.begin(), alpha_list.end(), std::round(anharm));
    int index = std::distance(alpha_list.begin(), iter);
    auto& error_arr = error_arr_[index];

    // func = CubicInterpolation2D(detune_list, mu_list, error_arr);
    func = LinearInterpolation2D(mu_list, detune_list, error_arr);
}

XTalkError1q::XTalkError1q(const std::vector<double>& alpha_list_,
    const std::vector<double>& mu_list_, 
    const std::vector<double>& detune_list_, 
    const std::vector<std::vector<std::vector<double>>> error_arr_,
    double anharm_)
{
    _init(alpha_list_, mu_list_, detune_list_, error_arr_, anharm_);
}

double XTalkError1q::singq_xtalk_err(double a, double detune, double mu) const
{
    profiler _("singq_xtalk_err");
    if (mu < 0.001) return 0;
    if (detune > 299.9999 || detune <= -499.9999) return 0;
    double error = a * func(mu, detune);
    return error;
}

double ResidualError1q::singq_residual_error(double a, double gamma, double fi, double fj, double alpha1, double alpha2) const
{
    profiler _("singq_residual_error");
    double error = lorentzain(fi, fj, a, gamma) + lorentzain(fi + alpha1, fj, a, gamma) + lorentzain(fi, fj + alpha2, a, gamma);
    if (error < 0)
        ThrowRuntimeError(fmt::format("error = {}", error));
    return error;
}


void T1Error2q::_init(const std::vector<double>& f_list_, const std::vector<double>& t1_list_)
{
    f_list = f_list_;
    t1_list = t1_list_;
    // func_interp = CubicSpline(f_list_, t1_list_);
    func_interp = Interp1d(f_list_, t1_list_);
}

T1Error2q::T1Error2q(const std::vector<double>& f_list_, const std::vector<double>& t1_list_)
    :f_list(f_list), t1_list(t1_list), func_interp(f_list, t1_list)
{
}

double T1Error2q::twoq_T1_err(double f_work, double f_idle, double a, double tq, const AcSpectrumParam& ac_spectrum_params) const
{
    int step = 1000;
    std::vector<double> ft = twoq_pulse(f_work, f_idle, tq, step, ac_spectrum_params);
    double error_sum = 0;

    for (auto f : ft)
    {
        double error = a * tq / func_interp(f);
        if (error < 0)
            error = 5.0e-4;
        error_sum += error;
    }

    return error_sum / step;
}

void T2Error2q::_init(const std::vector<double>& f_list_, const std::vector<double>& t2_list_)
{
    if (f_list.size() != 0)
    {
        f_list = f_list_;
        t2_list = t2_list_;
        use_dummy_t2_spectrum = false;
        func_interp = CubicSpline(f_list, t2_list);
    }
    else
    {
        use_dummy_t2_spectrum = true;
    }
}

T2Error2q::T2Error2q(const std::vector<double>& f_list_, const std::vector<double>& t2_list_)
{
    _init(f_list_, t2_list_);
}

double T2Error2q::twoq_T2_err(double f_work, double f_idle, double a, double tq, const AcSpectrumParam& ac_spectrum_params) const
{
    int step = 1000;
    std::vector<double> ft = twoq_pulse(f_work, f_idle, tq, step, ac_spectrum_params);
    double error_sum = 0;

    if (t2_list.size() > 0)
    {
        for (auto f : ft)
        {
            error_sum += func_interp(f);
        }
    }
    else
    {
        for (auto f : ft)
        {
            error_sum += T2Spectrum().singq_T2_err(a, tq, f, ac_spectrum_params);
        }
    }
    return a * error_sum * tq / step;
}

double XTalkError2q::twoq_xtalk_err(const std::pair<double, double>& fi, 
    const std::pair<double, double>& fj, 
    const double *a,
    double tq,
    double anharm1, double anharm2, 
    const AcSpectrumParam& ac_spectrum_param1, 
    const AcSpectrumParam& ac_spectrum_param2) const
{
    int step = 100;
    auto fits = twoq_pulse(fi.first, fi.second, tq, step, ac_spectrum_param1);
    auto fjts = twoq_pulse(fj.first, fj.second, tq, step, ac_spectrum_param2);

    double error_sum = 0;
    for (int i = 0; i < step; ++i)
    {
        error_sum += lorentzain(fits[i], fjts[i], a[0] , a[1]);
        error_sum += lorentzain(fits[i] + anharm1, fjts[i], a[2], a[3]);
        error_sum += lorentzain(fits[i], fjts[i] + anharm2, a[2], a[3]);
    }
    return error_sum * tq / step;
}

double PulseDistortion2q::twoq_pulse_distort_err(const std::pair<double, double>& fi, 
    const std::pair<double, double>& fj, double a,
    const AcSpectrumParam& ac_spectrum_paras1,
    const AcSpectrumParam& ac_spectrum_paras2) const
{
    double vi0 = freq2amp_formula(fi.first, ac_spectrum_paras1, false);
    double vi1 = freq2amp_formula(fi.second, ac_spectrum_paras1, false);
    double vj0 = freq2amp_formula(fj.first, ac_spectrum_paras2, false);
    double vj1 = freq2amp_formula(fj.second, ac_spectrum_paras2, false);

    return a * (std::abs(vi0 - vi1) + std::abs(vj0 - vj1));
}

/* To locate each qubit (i,j), the index should be i * W + j */
int ChipError::qubit_name_idx(int i, int j) const
{
    return i * W + j + 1;
}

int ChipError::qubit_idx(int i, int j) const
{
    return i * W + j;
}

std::pair<int, int> ChipError::qubit_idx_to_pos(int qubit_idx) const
{
    int i = qubit_idx / W;
    int j = qubit_idx % W;
    return { i, j };
}

bool ChipError::check_available_qubit_pos(int i, int j) const
{
    if (i < 0) return false;
    if (j < 0) return false;
    if (i >= H) return false;
    if (j >= W) return false;
    return true;
}

std::vector<std::pair<int, int>> ChipError::get_neighbors(int i, int j) const
{
    std::vector<std::pair<int, int>> ret;
    {
        int i_ = i;
        int j_ = j - 1;
        if (check_available_qubit_pos(i_, j_)) 
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i;
        int j_ = j + 1;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i - 1;
        int j_ = j;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i + 1;
        int j_ = j;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    return ret;
}

std::vector<std::pair<int, int>> ChipError::get_neighbors_distance_sqrt2(int i, int j) const
{
    std::vector<std::pair<int, int>> ret;
    {
        int i_ = i - 1;
        int j_ = j - 1;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i - 1;
        int j_ = j + 1;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i + 1;
        int j_ = j - 1;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i + 1;
        int j_ = j + 1;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    return ret;
}

std::vector<std::pair<int, int>> ChipError::get_neighbors_distance_2(int i, int j) const
{
    std::vector<std::pair<int, int>> ret;
    {
        int i_ = i - 2;
        int j_ = j;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i + 2;
        int j_ = j;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i;
        int j_ = j - 2;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    {
        int i_ = i;
        int j_ = j + 2;
        if (check_available_qubit_pos(i_, j_))
            ret.emplace_back(i_, j_);
    }
    return ret;
}

int ChipError::get_distance(int q1, int q2) const
{
    auto&& [q1_i, q1_j] = qubit_idx_to_pos(q1);
    auto&& [q2_i, q2_j] = qubit_idx_to_pos(q2);
    return L1norm(q1_i, q1_j, q2_i, q2_j);
}

void ChipError::_load_qubit_data(const rapidjson::Document& d)
{
    // H is 8
    // W is 6 (fixed)

    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            auto& node = qubits[qubit_idx(i, j)];
            node.x = i;
            node.y = j;
            
            auto& allow_freq = node.allow_freq;
            auto& isolated_error = node.isolated_error;
            auto& ac_spectrum = node.ac_spectrum;
            auto& t1_freq = node.t1_freq;
            auto& t1_t1 = node.t1_t1;
            auto& xy_crosstalk = node.xy_crosstalk;
            auto& anharm = node.anharm;

            int qubit_name = qubit_name_idx(i, j);
            if (!_load_qubit_info_from_json(d, qubit_name, allow_freq, isolated_error,
                ac_spectrum, t1_freq, t1_t1, xy_crosstalk, anharm))
            {
                node.used = false;
                fmt::print("q{} not found, skipped.\n", qubit_name);
            }
            else
            {
                node.used = true;
            }
        }
    }
}

void ChipError::_load_xy_crosstalk_sim_data(const rapidjson::Document& d)
{
    if (!get_alpha_list(d, alpha_list))
        ThrowRuntimeError(fmt::format("alpha_list has error in handling json."));

    if (!get_mu_list(d, mu_list))
        ThrowRuntimeError(fmt::format("mu_list has error in handling json."));

    if (!get_detune_list(d, detune_list))
        ThrowRuntimeError(fmt::format("detune_list has error in handling json."));

    if (!get_error_arr(d, error_arr))
        ThrowRuntimeError(fmt::format("error_arr has error in handling json."));
}

void ChipError::assemble_nodes()
{
    /* Use the data loaded from chipdata to assemble the qubits */
    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            _assemble_node(i, j);
        }
    }
}

void ChipError::_assemble_node(int i, int j)
{
    int qubit_index = qubit_idx(i, j);
    auto &node = qubits[qubit_index];
    node.frequency = -1.0; // frequency assigned later (a placeholder)
    if (!node.used)
    {
        // not in the available list, skip!
        return;
    }
    n_available_qubits++;
    // x is assigned in _load_qubit_data
    // y is assigned in _load_qubit_data
    node.t_sq = 20;
    // anharm is assigned in _load_qubit_data
    // sweet_point is no longer used
    // allow_freq is assigned in _load_qubit_data
    // bad_freq_range is assigned in _load_qubit_data
    node._preprocess_frequency(); // preprocess the frequency range to allow fast access
    // ac_spectrum is assigned in _load_qubit_data
    // t1_freq is assigned in _load_qubit_data
    // t1_t1 is assigned in _load_qubit_data
    // xy_crosstalk is assigned in _load_qubit_data
    node.freq_max = node.ac_spectrum.freq_max;
    node.freq_min = node.ac_spectrum.freq_min;

    /* Assemble model here */
    // T1_error removed.   node.model.t1_spectrum._init(node.t1_freq, node.t1_t1);
    // T2_error removed.   node.model.t2_spectrum._init({}, {}, node.ac_spectrum);
    node.model.xtalk_error._init(alpha_list, mu_list, detune_list, error_arr, node.anharm);
    /* residual error does not require to be assembled. */
    node.model.isolated_error._init(node.allow_freq_int, node.isolated_error);
}

int ChipError::from_qubits_to_coupler_idx(int q1, int q2) const
{
    if (q1 > q2) { std::swap(q1, q2); }
    if (q1 < 0 || q2 >= qubits.size())
        throw std::invalid_argument("qubit idx is invalid (out of range).");
    
    auto&& [q1_i, q1_j] = qubit_idx_to_pos(q1);
    auto&& [q2_i, q2_j] = qubit_idx_to_pos(q2);

    if ((q1_i == q2_i) && (q2_j - q1_j == 1))
    { 
        // in the same row, thus a vertical coupler        
        return q1_i * (2 * W - 1) + q1_j;
    }

    if ((q1_j == q2_j) && (q2_i - q1_i == 1))
    {
        // in the same column, thus a horizontal coupler        
        return q1_i * (2 * W - 1) + W - 1 + q1_j;
    }
    return -1;    
}

int ChipError::from_qubit_pos_to_coupler_idx(const std::pair<int, int>& q1, const std::pair<int, int>& q2) const
{
    int q1_idx = qubit_idx(q1.first, q1.second);
    int q2_idx = qubit_idx(q2.first, q2.second);
    
    return from_qubits_to_coupler_idx(q1_idx, q2_idx);
}

std::pair<int, int> ChipError::from_coupler_idx_to_qubits(int coupler_idx) const
{
    if (coupler_idx < 0 || coupler_idx >= couplers.size())
        throw std::invalid_argument("Coupler_idx is invalid (out of range).");

    // group_count
    int group_count = 2 * W - 1;
    int i = coupler_idx / group_count;
    int j = coupler_idx % group_count;

    if (j < (W - 1))
    {
        return { qubit_idx(i, j), qubit_idx(i, j + 1) };
    }
    else
    {
        // i+1, j may not be valid when i==H-1.
        j -= (W - 1);
        return { qubit_idx(i, j), qubit_idx(i + 1, j) };
    }
}

bool ChipError::check_available_coupler_pos(int direction, int row, int column) const
{
    if (row < 0) return false;
    if (column < 0) return false;

    if (direction == ChipCoupler::horizontal)
    {
        // row must be an even number
        if (row % 2) return false;

        // row/2 is the bus id, thus between 0~H-1
        if ((row / 2) >= H)
            return false;

        // column must be between 0~W-2
        if (column > (W - 2))
            return false;

        // otherwise it is a valid pos
        return true;
    }
    else if (direction == ChipCoupler::vertical)
    {
        // row must be an odd number
        if (!(row % 2)) return false;

        // row/2 is the bus id, but the last bus is not included, thus between 0~H-2
        if ((row / 2) >= (H - 1))
            return false;

        // column must be between 0~W-1
        if (column > (W - 1))
            return false;

        // otherwise it is a valid pos
        return true;
    }

    // the direction is not a valid number
    return false;
}

int ChipError::coupler_pos_to_coupler_id(int direction, int row, int column) const
{
    if (!check_available_coupler_pos(direction, row, column))
        throw std::runtime_error("Invalid position of a coupler");

    if (direction == ChipCoupler::vertical)
        column += (W - 1);

    return row / 2 + column;
}

std::vector<int> ChipError::get_neighbor_couplers(int coupler_id) const
{
    auto&& [q1, q2] = from_coupler_idx_to_qubits(coupler_id);

    auto&& [q1_i, q1_j] = qubit_idx_to_pos(q1);
    auto&& [q2_i, q2_j] = qubit_idx_to_pos(q2);

    // std::vector<int> overlapped;
    std::vector<int> nn_coupler;

    for (int i = 0; i < couplers.size(); ++i)
    {
        // fully overlapped
        if (i == coupler_id) continue;

        // if it is a broken coupler
        if (!couplers[i].used) continue;

        auto&& [q3, q4] = from_coupler_idx_to_qubits(i);

        auto&& [q3_i, q3_j] = qubit_idx_to_pos(q3);
        auto&& [q4_i, q4_j] = qubit_idx_to_pos(q4);

        // overlapped with any qubit
        if (q1 == q3 || q1 == q4 || q2 == q3 || q2 == q4)
        {
            // overlapped.push_back(i);
            continue;
        }

        if (L1norm(q1_i, q1_j, q3_i, q3_j) == 1 ||
            L1norm(q1_i, q1_j, q4_i, q4_j) == 1 ||
            L1norm(q2_i, q2_j, q3_i, q3_j) == 1 ||
            L1norm(q2_i, q2_j, q4_i, q4_j) == 1
            )
        {
            nn_coupler.push_back(i);
        }
    }

    return nn_coupler;
}

//std::vector<std::tuple<int, int, int>> ChipError::get_overlapped_couplers(int direction, int row, int column) const
//{
//    if (direction == ChipCoupler::horizontal)
//    {
//        return _get_overlapped_couplers_horizontal(row, column);
//    }
//    else if (direction == ChipCoupler::vertical)
//    {
//        return _get_overlapped_couplers_vertical(row, column);
//    }
//    throw std::runtime_error("Bad position input.");    
//}
//
//std::vector<std::tuple<int, int, int>> ChipError::_get_overlapped_couplers_horizontal(int row, int column) const
//{
//    std::vector<std::tuple<int, int, int>> ret;
//    ret.reserve(6);
//    // the same direction, column +- 1
//    {
//        int direction_ = ChipCoupler::horizontal;
//        int row_ = row;
//        int column_ = column - 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::horizontal;
//        int row_ = row;
//        int column_ = column + 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    // different direction, row +- 1, column/column+1
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row - 1;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row - 1;
//        int column_ = column + 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row + 1;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row + 1;
//        int column_ = column + 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    return ret;
//}
//
//std::vector<std::tuple<int, int, int>> ChipError::_get_overlapped_couplers_vertical(int row, int column) const
//{
//    std::vector<std::tuple<int, int, int>> ret;
//    ret.reserve(6);
//    // the same direction, row +- 2
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row - 2;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row + 2;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    // different direction, row +- 1, column - 1/column
//    {
//        int direction_ = ChipCoupler::horizontal;
//        int row_ = row - 1;
//        int column_ = column - 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::horizontal;
//        int row_ = row - 1;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::horizontal;
//        int row_ = row + 1;
//        int column_ = column - 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::horizontal;
//        int row_ = row + 1;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    return ret;
//}
//
//std::vector<std::tuple<int, int, int>> ChipError::get_distance_1_couplers(int direction, int row, int column) const
//{
//    if (direction == ChipCoupler::horizontal)
//    {
//        return _get_distance_1_couplers_horizontal(row, column);
//    }
//    else if (direction == ChipCoupler::vertical)
//    {
//        return _get_distance_1_couplers_vertical(row, column);
//    }
//    throw std::runtime_error("Bad position input.");
//}
//
//std::vector<std::tuple<int, int, int>> ChipError::_get_distance_1_couplers_horizontal(int row, int column) const
//{
//    std::vector<std::tuple<int, int, int>> ret;
//    ret.reserve(2);
//    // the same direction, row +- 2
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row - 2;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row + 2;
//        int column_ = column;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    return ret;
//}
//
//std::vector<std::tuple<int, int, int>> ChipError::_get_distance_1_couplers_vertical(int row, int column) const
//{
//    std::vector<std::tuple<int, int, int>> ret;
//    ret.reserve(2);
//    // the same direction, column +- 1
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row;
//        int column_ = column - 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    {
//        int direction_ = ChipCoupler::vertical;
//        int row_ = row;
//        int column_ = column + 1;
//        if (check_available_coupler_pos(direction_, row_, column_))
//            ret.emplace_back(direction_, row_, column_);
//    }
//    return ret;
//}

void ChipError::assemble_couplers()
{
    for (int i = 0; i < couplers.size(); ++i)
    {
        _assemble_coupler(i);
    }
    for (int i = 0; i < couplers.size(); ++i)
    {
        couplers[i].neighbor_couplers = get_neighbor_couplers(i);
    }
}

void ChipError::_assemble_coupler(int coupler_id)
{
    auto&& [q1, q2] = from_coupler_idx_to_qubits(coupler_id);
    auto&& [q1_i, q1_j] = qubit_idx_to_pos(q1);
    auto&& [q2_i, q2_j] = qubit_idx_to_pos(q2);
    
    auto& coupler = couplers[coupler_id];
    coupler.frequency = -1;

    if (!(qubits[q1].used && qubits[q2].used))
    {
        // if any of q1 and q2 is not used
        coupler.used = false;
        return;
    }
    coupler.used = true;
    coupler.qubit1 = q1;
    coupler.qubit2 = q2;
    coupler.coupler_id = coupler_id;

    coupler.initialize_direction_and_pos(H, W);

    n_available_couplers++;
    coupler.t_twoq = 40;

    coupler.model.t1_error_q1._init(qubits[q1].t1_freq, qubits[q1].t1_t1);
    coupler.model.t1_error_q2._init(qubits[q2].t1_freq, qubits[q2].t1_t1);
    coupler.model.t2_error_q1._init({}, {});
    coupler.model.t2_error_q2._init({}, {});
}

/* Initialize all qubits to unallocated case */

void ChipError::initialize_all_qubits()
{
    for (auto& node : qubits)
        node.frequency = -1.0;
}

/* List all unallocated qubits, return in a list of (x,y) pairs */

std::vector<std::pair<int, int>> ChipError::list_all_unallocated_qubits() const
{
    std::vector<std::pair<int, int>> unallocated_nodes;
    for (auto& node : qubits)
    {
        if (node.frequency < 0)
            unallocated_nodes.emplace_back(node.x, node.y);
    }
    return unallocated_nodes;
}

/* List all unallocated qubits, return in a list of (x,y) pairs */

std::vector<std::pair<int, int>> ChipError::list_all_allocated_qubits() const
{
    std::vector<std::pair<int, int>> allocated_nodes;
    for (auto& node : qubits)
    {
        if (node.frequency >= 0)
            allocated_nodes.emplace_back(node.x, node.y);
    }
    return allocated_nodes;
}

void ChipError::assign_qubit_frequencies(const std::vector<double>& frequencies)
{
    int i = 0;
    for (auto& node : qubits)
    {
        if (!node.used) continue;
        node.frequency = frequencies[i];
        i++;
    }
}

void ChipError::assign_qubit_frequencies_with_ranges(const std::vector<double>& ranges)
{
    int i = 0;
    for (auto& node : qubits)
    {
        if (!node.used) continue;
        node.assign_frequency_on_range(ranges[i]);
        i++;
    }
}

void ChipError::assign_qubit_frequencies_full(const std::vector<double>& frequencies)
{
    int i = 0;
    if (qubits.size() != frequencies.size())
    {
        throw std::runtime_error("Size not match (in assign_qubit_frequencies_full).");
    }
    for (auto& node : qubits)
    {
        node.frequency = frequencies[i];
        i++;
    }
}

void ChipError::assign_qubit_frequencies_by_idx_dict(const std::map<int, double>& frequencies)
{
    initialize_all_qubits();
    for (auto&& [qubit_index, freq] : frequencies)
    {
        qubits[qubit_index].frequency = freq;
    }
}

void ChipError::assign_qubit_frequencies_by_pair_dict(const std::map<std::pair<int, int>, double>& frequencies)
{
    initialize_all_qubits();
    for (auto&& [qubit_pos, freq] : frequencies)
    {
        qubits[qubit_idx(qubit_pos.first, qubit_pos.second)].frequency = freq;
    }
}

std::vector<std::pair<double, double>> ChipError::list_qubit_freq_ranges() const
{
    std::vector<std::pair<double, double>> ret;
    for (auto& node : qubits)
    {
        if (!node.used) continue;
        ret.emplace_back(node.allow_freq[0], node.allow_freq.back());
    }
    return ret;
}

bool ChipError::unused_or_unallocated_qubit(int qubit_index) const
{
    auto& node = qubits[qubit_index];

    // used and allocated
    if (node.frequency >= 0 && node.used)
        return false;
    else
        return true;
}

bool ChipError::unused_or_unallocated_qubit(int i, int j) const
{
    int qubit_index = qubit_idx(i, j);
    return unused_or_unallocated_qubit(qubit_index);
}

void ChipError::initialize_all_couplers()
{
    for (auto& coupler : couplers)
    {
        coupler.frequency = -1;
    }
}

std::vector<int> ChipError::list_all_unallocated_couplers() const
{
    std::vector<int> coupler_ids;
    for (auto& coupler : couplers)
    {
        if (coupler.used && coupler.frequency <= 0)
            coupler_ids.push_back(coupler.coupler_id);
    }
    return coupler_ids;
}

std::vector<int> ChipError::list_all_allocated_couplers() const
{
    std::vector<int> coupler_ids;
    for (auto& coupler : couplers)
    {
        if (coupler.used && coupler.frequency > 0)
            coupler_ids.push_back(coupler.coupler_id);
    }
    return coupler_ids;
}

void ChipError::assign_coupler_frequencies(const std::vector<double>& frequencies)
{
    int i = 0;
    if (couplers.size() != frequencies.size())
    {
        throw std::runtime_error("Size not match (in assign_coupler_frequencies).");
    }
    for (auto& coupler : couplers)
    {
        coupler.frequency = frequencies[i];
        i++;
    }
}

void ChipError::assign_coupler_frequencies_by_idx_dict(const std::map<int, double>& frequencies)
{
    initialize_all_couplers();
    for (auto&& [coupler_idx, freq] : frequencies)
    {
        couplers[coupler_idx].frequency = freq;
    }
}

void ChipError::assign_coupler_frequencies_by_pair_dict(const std::map<std::pair<std::pair<int, int>, std::pair<int, int>>, double>& frequencies)
{
    initialize_all_couplers();
    for (auto&& [coupler_q1q2, freq] : frequencies)
    {
        int coupler_idx = from_qubit_pos_to_coupler_idx(coupler_q1q2.first, coupler_q1q2.second);
        couplers[coupler_idx].frequency = freq;
    }
}

bool ChipError::_unused_or_unallocated_coupler(int q1, int q2, int coupler_id) const
{
    auto&& [q1_i, q1_j] = qubit_idx_to_pos(q1);
    auto&& [q2_i, q2_j] = qubit_idx_to_pos(q2);

    /* When any qubit is not allocated, then the coupler is not considered */
    if (unused_or_unallocated_qubit(q1_i, q1_j)) return true;
    if (unused_or_unallocated_qubit(q2_i, q2_j)) return true;

    if (!couplers[coupler_id].used) return true;
    if (couplers[coupler_id].frequency < 0) return true;

    return false;
}

bool ChipError::unused_or_unallocated_coupler(int q1, int q2) const
{
    int coupler_id = from_qubits_to_coupler_idx(q1, q2);
    return _unused_or_unallocated_coupler(q1, q2, coupler_id);
}

bool ChipError::unused_or_unallocated_coupler(int coupler_id) const
{
    auto&& [q1, q2] = from_coupler_idx_to_qubits(coupler_id);
    return _unused_or_unallocated_coupler(q1, q2, coupler_id);
}


void ChipError::_read_file_to_json(const char* filename, rapidjson::Document& d)
{
    FILE* fp = fopen(filename, "r");
    if (!fp)
        ThrowInvalidArgument(fmt::format("Unable to open file: {}", filename));

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    d.ParseStream(is);
    fclose(fp);
}

void ChipError::load_file(const char* qubit_data_filename, const char* xy_crosstalk_sim_filename)
{
    {
        rapidjson::Document d;
        _read_file_to_json(qubit_data_filename, d);
        _load_qubit_data(d);
        fmt::print("qubit_data loaded\n");
        // load_xy_crosstalk(d, xy_crosstalk);
        fmt::print("xy_crosstalk loaded\n");
    }
    {
        rapidjson::Document d;
        _read_file_to_json(xy_crosstalk_sim_filename, d);
        _load_xy_crosstalk_sim_data(d);
        fmt::print("xy_crosstalk_sim loaded\n");
    }
    assemble_nodes();
    assemble_couplers();
}

ChipError::ChipError(int H_, int W_)
    : H(H_), W(W_)
{
    // generate the containers
    qubits.resize(H * W);

    // W=6, H=12
    // 2*W-1 = 11, H-1=11, W-1=5, 11*11+5=126
    int coupler_count = (2 * W - 1) * (H - 1) + W - 1;

    couplers.resize(coupler_count);
}

inline void ChipQubit::_preprocess_frequency()
{
    // std::sort(allow_freq.begin(), allow_freq.end());
    
    allow_freq_int.resize(allow_freq.size());
    for (int i = 0; i < allow_freq.size(); ++i)
    {
        allow_freq_int[i] = int(allow_freq[i]);
    }

    // reverse allow_freq, isolated_error

    std::reverse(allow_freq_int.begin(), allow_freq_int.end());
    std::reverse(allow_freq.begin(), allow_freq.end());
    std::reverse(isolated_error.begin(), isolated_error.end());

    // std::sort(bad_freq_range.begin(), bad_freq_range.end());
     
    /* subtract 1e-4 to make sure allow_freq[0] can be contained */
    //frequency_ranges.push_back(allow_freq[0] - 1e-4);
    //for (auto&& [bad_range_l, bad_range_r] : bad_freq_range)
    //{
    //    frequency_ranges.push_back(bad_range_l);
    //    frequency_ranges.push_back(bad_range_r);
    //}
    ///* add 1e-4 to make sure allow_freq[-1] can be contained */
    //frequency_ranges.push_back(allow_freq.back() + 1e-4);

    //// Ensure every allow_freq can pass the check!
    //for (auto freq : allow_freq)
    //{
    //    frequency = freq;
    //    if (!frequency_in_allow_range())
    //        ThrowRuntimeError("Allow frequency has conflict with bad_freq_range");
    //}
    //frequency = -1.0;
}

bool ChipQubit::frequency_in_allow_range() const
{
    // search for the first range fi such that freq <= fi
    // it means [f_{i-1}, f_{i}] contains freq
    // when i is even, it means freq in bad range
    // when i is odd, it means freq in good range
    // 
    // *************** OLD VERSION **********************
    //auto iter = std::lower_bound(frequency_ranges.begin(), frequency_ranges.end(), frequency);
    //if (iter == frequency_ranges.end())
    //    return false;

    //int distance = std::distance(frequency_ranges.begin(), iter);
    //if (distance & 1)
    //{
    //    // when distance is 1,3,5,...
    //    return true;
    //}
    //else
    //{
    //    // when distance is 0,2,4,...
    //    return false;
    //}

    int frequency_round = int(std::round(frequency));

    // the result for binary search is whether the frequency is allowed
    return std::binary_search(allow_freq_int.begin(), allow_freq_int.end(), frequency_round);

}

void IsolatedError1q::_init(const std::vector<int>& f_list_,
    const std::vector<double>& isolated_error_)
{
    f_list = f_list_;
    isolated_error_list = isolated_error_;
}

double IsolatedError1q::isolated_error(double f) const
{
    int f_int = int(f);
    auto iter = std::lower_bound(f_list.begin(), f_list.end(), f_int);
    if (iter == f_list.end())
        return 1.0;

    int pos = std::distance(f_list.begin(), iter);
    return isolated_error_list[pos];
}

void ChipCoupler::initialize_direction_and_pos(int H, int W)
{
    // determine the direction (vertical, horizontal)
    int pos_in_group = coupler_id % (2 * W - 1);
    if (pos_in_group >= (W - 1))
        // W-1 ~ 2*W-2
        direction = vertical;
    else
        // 0 ~ W-2
        direction = horizontal;

    int group_id = coupler_id / (2 * W - 1);

    // determine the row_id
    row = group_id * 2;

    if (direction == vertical)
        row += 1;

    // determine the column_id
    column = pos_in_group;
    if (direction == vertical)
        column -= (W - 1);
}
