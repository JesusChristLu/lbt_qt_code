#pragma once
#include <vector>
#include <map>
#include "interpolation.h"
#include "formula.h"
#include "load_json.h"

struct T1Spectrum
{
    std::vector<double> f_list;
    std::vector<double> t1_list;
    // CubicSpline func_interp;
    Interp1d func_interp;

    T1Spectrum() {}
    void _init(const std::vector<double>& f_list_, const std::vector<double>& t1_list_);
    T1Spectrum(const std::vector<double>& f_list_, const std::vector<double>& t1_list_);
    double singq_T1_err(double a, double tq, double freq) const;
};

struct T2Spectrum
{
    std::vector<double> f_list;
    std::vector<double> t2_list;
    bool use_dummy_t2_spectrum = false;
    CubicSpline func_interp;

    T2Spectrum() {}

    void _init(const std::vector<double>& f_list_,
        const std::vector<double>& t2_list_);

    T2Spectrum(const std::vector<double>& f_list_,
        const std::vector<double>& t2_list_);

    double singq_T2_err(double a, double tq, double f, const AcSpectrumParam& params) const;
};

struct IsolatedError1q
{
    std::vector<int> f_list;
    std::vector<double> isolated_error_list;

    IsolatedError1q() {}

    void _init(const std::vector<int> &f_list_, const std::vector<double> &isolated_error_);

    double isolated_error(double f) const;
};

struct XTalkError1q
{
    std::vector<double> alpha_list;
    std::vector<double> mu_list;
    std::vector<double> detune_list;
    std::vector<std::vector<std::vector<double>>> error_arr;
    LinearInterpolation2D func;
    double anharm = 0.0;
    
    XTalkError1q() {}

    void _init(
        const std::vector<double>& alpha_list_, const std::vector<double>& mu_list_,
        const std::vector<double>& detune_list_,
        const std::vector<std::vector<std::vector<double>>> error_arr_,
        double anharm);

    XTalkError1q(
        const std::vector<double>& alpha_list_, const std::vector<double>& mu_list_,
        const std::vector<double>& detune_list_, 
        const std::vector<std::vector<std::vector<double>>> error_arr_,
        double anharm);

    double singq_xtalk_err(double a, double detune, double mu) const;
};

struct ResidualError1q
{
    double singq_residual_error(double a, double gamma, double fi, double fj, double alpha1, double alpha2) const;
};

struct NodeErrorModel
{
    // T1Spectrum t1_spectrum;
    // T2Spectrum t2_spectrum;
    IsolatedError1q isolated_error;
    XTalkError1q xtalk_error;
    ResidualError1q residual_error;

    NodeErrorModel() {}

    NodeErrorModel(
        const std::vector<double> &t1_flist_,
        const std::vector<double> &t1_t1_list_,
        const std::vector<double> &ac_spectrum_,
        const std::vector<double> &alpha_list,
        const std::vector<double> &mu_list,
        const std::vector<double> &detune_list,
        const std::vector<std::vector<std::vector<double>>> error_arr_,
        double anharm) :
        // t1_spectrum(t1_flist_, t1_t1_list_),
        // t2_spectrum({}, {}, ac_spectrum_),
        xtalk_error(alpha_list, mu_list, detune_list, error_arr_, anharm)
    {}
};

struct T1Error2q
{
    std::vector<double> f_list;
    std::vector<double> t1_list;
    // CubicSpline func_interp;
    Interp1d func_interp;

    T1Error2q() {}
    void _init(const std::vector<double>& f_list_, const std::vector<double>& t1_list_);
    T1Error2q(const std::vector<double>& f_list_, const std::vector<double>& t1_list_);
    double twoq_T1_err(double f_work , double f_idle, double a, double tq, 
        const AcSpectrumParam& ac_spectrum_params) const;

};

struct T1Error2qSimplified
{
    T1Spectrum t1_error;
    T1Error2qSimplified() {}
    inline void _init(const std::vector<double>& f_list_, const std::vector<double>& t1_list_)
    {
        t1_error._init(f_list_, t1_list_);
    }
    inline double twoq_T1_err(double freq, double a, double tq) const
    {
        return t1_error.singq_T1_err(a, tq, freq);
    }
};

struct T2Error2q
{
    std::vector<double> f_list;
    std::vector<double> t2_list;
    bool use_dummy_t2_spectrum = false;
    CubicSpline func_interp;

    T2Error2q() {}

    void _init(const std::vector<double>& f_list_,
        const std::vector<double>& t2_list_);

    T2Error2q(const std::vector<double>& f_list_,
        const std::vector<double>& t2_list_);

    double twoq_T2_err(double f_work, double f_idle, double a, double tq, 
        const AcSpectrumParam& ac_spectrum_params) const;

};

struct T2Error2qSimplified
{
    T2Spectrum t2_error;

    T2Error2qSimplified() {}
    inline void _init(const std::vector<double>& f_list_, const std::vector<double>& t1_list_)
    {
        t2_error._init(f_list_, t1_list_);
    }
    inline double twoq_T2_err(double freq, double a, double tq, const AcSpectrumParam& ac_spectrum) const
    {
        return t2_error.singq_T2_err(a, tq, freq, ac_spectrum);
    }
};

struct XTalkError2q
{
    double twoq_xtalk_err(const std::pair<double, double>& fi,
        const std::pair<double, double>& fj, 
        const double* a, double tq,
        double anharm1, double anharm2,
        const AcSpectrumParam& ac_spectrum_param1,
        const AcSpectrumParam& ac_spectrum_param2) const;
};

struct XTalkError2qSimplified
{
    inline double twoq_xtalk_err(double pulse1, double pulse2,
        const double* a, double anharm1, double anharm2) const
    {
        double error = 0;
        error += lorentzain(pulse1, pulse2, a[0], a[1]);
        error += lorentzain(pulse1 + anharm1, pulse2, a[2], a[3]);
        error += lorentzain(pulse1, pulse2 + anharm2, a[2], a[3]);
        return error;
    }
};

struct PulseDistortion2q
{
    double twoq_pulse_distort_err(
        const std::pair<double, double>& fi,
        const std::pair<double, double>& fj,
        double a,
        const AcSpectrumParam& ac_spectrum_paras1,
        const AcSpectrumParam& ac_spectrum_paras2) const;
};

struct InnerLeakage
{
    double inner_leakage(const std::pair<double, double>& f1, const std::pair<double, double>& f2, double a1, double a2) const
    {
        //auto [min_f1, max_f1] = std::minmax(f1.first, f1.second);
        //auto [min_f2, max_f2] = std::minmax(f2.first, f2.second);
        //if ((min_f1 > max_f2 && max_f1 > min_f2) || (min_f2 > max_f1 && max_f2 > min_f1))
        //    return a1;
        //else
        //    return a2;
        return 0;
    }
};

struct CouplerErrorModel
{
    // T1Error2q t1_error_q1;
    // T1Error2q t1_error_q2;
    // T2Error2q t2_error_q2;
    // T2Error2q t2_error_q1;
    T1Error2qSimplified t1_error_q1;
    T1Error2qSimplified t1_error_q2;
    T2Error2qSimplified t2_error_q1;
    T2Error2qSimplified t2_error_q2;
    // XTalkError2q xtalk_error;
    XTalkError2qSimplified xtalk_error;
    PulseDistortion2q distort_error;
    InnerLeakage inner_leakage_error;
};

struct ChipQubit
{
    bool used;
    int x;
    int y;
    double frequency;
    double t_sq;
    double anharm;
    // double sweet_point;
    std::vector<double> allow_freq;
    std::vector<double> isolated_error;

    std::vector<int> allow_freq_int;
    AcSpectrumParam ac_spectrum;
    std::vector<double> t1_freq;
    std::vector<double> t1_t1;
    std::vector<double> xy_crosstalk;
    NodeErrorModel model;

    /* Min max frequency from ac spectrum */
    double freq_min;
    double freq_max;
    
    /* preprocess to allow fast calculation for frequency range */
    void _preprocess_frequency();

    /* use binary search to locate if freq in any good range */
    bool frequency_in_allow_range() const;

    inline void assign_frequency(double frequency_) 
    { 
        frequency = frequency_; 
    }
    inline void assign_frequency_on_range(double range) 
    { 
        int allow_freq_count = allow_freq.size();
        int allow_index = std::round((allow_freq_count - 1) * range);
        frequency = allow_freq[allow_index];
    }
};

struct ChipCoupler
{
    bool used;
    int qubit1;
    int qubit2;
    int coupler_id;
    std::vector<int> neighbor_couplers;

    constexpr static int horizontal = 0; // (0-1)
    constexpr static int vertical = 1; // (0-6)

    int direction;
    int row;
    int column;

    void initialize_direction_and_pos(int H, int W);

    double t_twoq;
    double frequency;

    CouplerErrorModel model;

    inline void assign_frequency(double frequency_)
    {
        frequency = frequency_;
    }

};

struct ChipError
{
    ChipError(int H_, int W_);
    int H; // height
    int W; // width
    std::vector<ChipQubit> qubits;
    std::vector<ChipCoupler> couplers;
    int n_available_qubits = 0; /* initialize from assemble_node */
    int n_available_couplers = 0; /* initialize from assemble_node */

    /* Global data */
    std::vector<std::vector<double>> xy_crosstalk; /* from qubit_data */
    std::vector<std::vector<std::vector<double>>> error_arr; /* from xy_crosstalk_sim */
    std::vector<double> alpha_list; /* from xy_crosstalk_sim */
    std::vector<double> mu_list; /* from xy_crosstalk_sim */
    std::vector<double> detune_list; /* from xy_crosstalk_sim */

    /* To locate each qubit (i,j), the index should be i * W + j */
    int qubit_name_idx(int i, int j) const;
    /* Locate it in the vector */
    int qubit_idx(int i, int j) const;
    /* When i+j is odd ((i+j)%2==1), it is a high freq qubit. */
    // inline bool is_high_freq_qubit(int i, int j) const { return (i + j) % 2; }
    /* Compute i,j from qubit_idx */
    std::pair<int, int> qubit_idx_to_pos(int qubit_idx) const;

    bool check_available_qubit_pos(int i, int j) const;
    std::vector<std::pair<int, int>> get_neighbors(int i, int j) const;
    std::vector<std::pair<int, int>> get_neighbors_distance_sqrt2(int i, int j) const;
    std::vector<std::pair<int, int>> get_neighbors_distance_2(int i, int j) const;

    int get_distance(int q1, int q2) const;

    void load_file(const char* qubit_data_filename, const char* xy_crosstalk_sim_filename);
    void _read_file_to_json(const char* filename, rapidjson::Document& d);
    void _load_qubit_data(const rapidjson::Document& d);
    void _load_xy_crosstalk_sim_data(const rapidjson::Document& d);
    
    void assemble_nodes();
    void _assemble_node(int i, int j);

    /* Use an integer to locate each coupler */
    int from_qubits_to_coupler_idx(int q1, int q2) const;
    int from_qubit_pos_to_coupler_idx(const std::pair<int, int>& q1, const std::pair<int, int>& q2) const;
    std::pair<int, int> from_coupler_idx_to_qubits(int coupler_idx) const;
    bool check_available_coupler_pos(int direction, int row, int column) const;
    int coupler_pos_to_coupler_id(int direction, int row, int column) const;
    std::vector<int> get_neighbor_couplers(int coupler_id) const;
    //std::vector<std::tuple<int, int, int>> get_overlapped_couplers(int direction, int row, int column) const;
    //std::vector<std::tuple<int, int, int>> _get_overlapped_couplers_horizontal(int row, int column) const;
    //std::vector<std::tuple<int, int, int>> _get_overlapped_couplers_vertical(int row, int column) const;

    //std::vector<std::tuple<int, int, int>> get_distance_1_couplers(int direction, int row, int column) const;
    //std::vector<std::tuple<int, int, int>> _get_distance_1_couplers_horizontal(int row, int column) const;
    //std::vector<std::tuple<int, int, int>> _get_distance_1_couplers_vertical(int row, int column) const;

    void assemble_couplers();
    void _assemble_coupler(int coupler_id);

    /***************** For single gate allocation *****************/

    /* Initialize all qubits to unallocated case */
    void initialize_all_qubits();
    /* List all unallocated qubits, return in a list of (x,y) pairs */
    std::vector<std::pair<int, int>> list_all_unallocated_qubits() const;
    /* List all unallocated qubits, return in a list of (x,y) pairs */
    std::vector<std::pair<int, int>> list_all_allocated_qubits() const;
    void assign_qubit_frequencies(const std::vector<double>& frequencies);
    void assign_qubit_frequencies_with_ranges(const std::vector<double>& ranges);
    void assign_qubit_frequencies_full(const std::vector<double>& frequencies);
    void assign_qubit_frequencies_by_idx_dict(const std::map<int, double>& frequencies);
    void assign_qubit_frequencies_by_pair_dict(const std::map<std::pair<int, int>, double>& frequencies);
    std::vector<std::pair<double, double>> list_qubit_freq_ranges() const;
    bool unused_or_unallocated_qubit(int qubit_index) const;
    bool unused_or_unallocated_qubit(int i, int j) const;

    /***************** For two qubit gate allocation *****************/

    void initialize_all_couplers();
    std::vector<int> list_all_unallocated_couplers() const;
    std::vector<int> list_all_allocated_couplers() const;
    void assign_coupler_frequencies(const std::vector<double>& frequencies);
    void assign_coupler_frequencies_by_idx_dict(const std::map<int, double>& frequencies);
    void assign_coupler_frequencies_by_pair_dict(const std::map<std::pair<std::pair<int, int>, std::pair<int, int>>, double>& frequencies);

    bool _unused_or_unallocated_coupler(int q1, int q2, int coupler_id) const;
    bool unused_or_unallocated_coupler(int q1, int q2) const;
    bool unused_or_unallocated_coupler(int coupler_id) const;
};

struct FrequencyAllocator
{
    ChipError chip;
    inline FrequencyAllocator()
        : chip(12, 6)
    {
        /* Remember to execute load_file first */
    }
    inline static FrequencyAllocator& get_instance()
    {
        /* Make a singleton */
        static FrequencyAllocator instance;
        return instance;
    }
    inline static ChipError& get_chip()
    {
        return get_instance().chip;
    }
};