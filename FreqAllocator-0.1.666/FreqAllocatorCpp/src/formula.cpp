#include "formula.h"
#include "interpolation.h"

double lorentzain(double fi, double fj, double a, double gamma) {
    double wave = (1.0 / M_PI) * (gamma / ((fi - fj) * (fi - fj) + (gamma * gamma)));
    return a * wave;
}

// Define the T1_spectra function
std::vector<double> T1_spectra(double fMax, int step) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> randomDist(0.0, 1.0);

    int badFreqNum = static_cast<int>(randomDist(generator) * 5);
    std::vector<double> fList;
    for (int i = 0; i < step; i++) {
        double f = 3.75 + (fMax - 3.75) * i / (step - 1);
        fList.push_back(f);
    }

    double gamma = 1e-3 + (2e-2 - 1e-3) * randomDist(generator);
    std::vector<double> T1(step);
    for (int i = 0; i < step; i++) {
        T1[i] = std::normal_distribution<double>(static_cast<double>(std::rand() % 30 + 20), 5.0)(generator);
    }

    for (int _ = 0; _ < badFreqNum; _++) {
        double a = randomDist(generator) * 0.6;
        double badFreq = 3.75 + (fMax - 3.75) * randomDist(generator);
        for (int i = 0; i < step; i++) {
            T1[i] -= lorentzain(fList[i], badFreq, a, gamma);
        }
    }

    for (int i = 0; i < step; i++) {
        T1[i] = std::max(1.0, T1[i]);
    }

    std::vector<double> result(step);
    for (int i = 0; i < step; i++) {
        result[i] = 1e-3 / T1[i];
    }

    return result;
}

// Define the f_phi_spectra function
double f_phi_spectra(double fMax, double phi) {
    double d = 0.0;
    return fMax * sqrt(fabs(cos(M_PI * phi)) * sqrt(1 + d * d * tan(M_PI * phi) * tan(M_PI * phi)));
}

// Define the phi2f function
double phi2f(double phi, double fMax, int step) {
    std::vector<double> phiList(step);
    for (int i = 0; i < step; i++) {
        phiList[i] = i * 0.5 / (step - 1);
    }

    std::vector<double> fList(step);
    for (int i = 0; i < step; i++) {
        fList[i] = f_phi_spectra(fMax, phiList[i]);
    }
    CubicSpline func_interp(phiList, fList);
    return func_interp(phi);
}

// Define the f2phi function
double f2phi(double f, double fq_max, double Ec, double d, double w, double g) {
    f = f - g * g / (f - w);
    double alpha = (f + Ec) / (Ec + fq_max);
    double beta = (alpha * alpha * alpha * alpha - d * d) / (1 - d * d);
    beta = sqrt(beta);
    double phi = acos(beta);
    return phi;
}

// Define the f2phi function
double f2phi(double f, double fq_max, double Ec, double d) {    
    double alpha = (f + Ec) / (Ec + fq_max);
    double beta = (alpha * alpha * alpha * alpha - d * d) / (1 - d * d);
    beta = sqrt(beta);
    double phi = acos(beta);
    return phi;
}

double freq2amp_formula(double x, const AcSpectrumParam& params, bool tans2phi)
{
    if (params.w.has_value())
    {
        double g = params.g.value();
        double w = params.w.value();
        x = x - g * g / (x - w);
    }
    double detune = params.detune;
    double fq_max = params.fq_max;
    double d = params.d;
    // double offset = params.offset;
    double M = params.M;

    double alpha = (x + detune) / (detune + fq_max);
    double beta = (alpha * alpha * alpha * alpha - d * d) / (1 - d * d);

    if (beta < 0 || beta > 1)
        throw std::runtime_error(fmt::format("beta < 0 (in freq2amp). beta = {}", beta));
    
    double phi = std::abs(std::acos(std::sqrt(beta)));

    if (tans2phi)
        return phi;
    else{
        double amp = phi / (M * M_PI);
        return amp;
    }
}

double amp2freq_formula(double x, const AcSpectrumParam& params, bool tans2phi)
{
    double detune = params.detune;
    double fq_max = params.fq_max;
    double d = params.d;
    // double offset = params.offset;
    double M = params.M;

    double phi;
    if (tans2phi)
        phi = x;
    else
        phi = M_PI * M * x;

    double fq = (fq_max + detune) * std::sqrt(
        std::sqrt(1 + d * d * std::tan(phi) * std::tan(phi)) * std::abs(std::cos(phi))
    ) - detune;

    if (params.w.has_value())
    {
        double w = params.w.value();
        double g = params.g.value();
        double fg = std::sqrt((w - fq) * (w - fq) + 4 * g * g);
        fq = (w + fq + fg) / 2;
    }

    return fq;
}

std::vector<double> twoq_pulse(double fwork, double fidle, double tq, int step, const AcSpectrumParam& ac_spectrum_param)
{
    if (fwork == fidle)
    {
        return std::vector<double>(step, fwork);
    }

    double VIdle = freq2amp_formula(fidle, ac_spectrum_param, false);
    double VWork = freq2amp_formula(fwork, ac_spectrum_param, false);
    auto pulselen = tq;
    double sigma = 1.5;
    double flattop_start = 3 * sigma;
    double flattop_end = pulselen - 3 * sigma;

    std::vector<double> freq_list(step);
    for (int i = 0; i < step; ++i)
    {
        double t = pulselen / (step - 1) * i;
        double v = (VWork - VIdle) / 2 *
            std::erf((t - flattop_start) / (std::sqrt(2) * sigma)) -
            std::erf((t - flattop_end) / (std::sqrt(2) * sigma)) + VIdle;
        freq_list[i] = amp2freq_formula(v, ac_spectrum_param, false);
    }

    return freq_list;
}
