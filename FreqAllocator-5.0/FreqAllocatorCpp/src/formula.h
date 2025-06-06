#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "utils.h"

#include "interpolation.h"

double lorentzain(double fi, double fj, double a, double gamma);

std::vector<double> T1_spectra(double fMax, int step);

double f_phi_spectra(double fMax, double phi);

double phi2f(double phi, double fMax, int step);

double f2phi(double f, double fq_max, double Ec, double d, double w, double g);

double f2phi(double f, double fq_max, double Ec, double d);

struct AcSpectrumParam
{
    //enum BranchEnum
    //{
    //    Right,
    //    Left,
    //    BranchUndefined,
    //};
    //enum SpectrumEnum
    //{
    //    Standard,
    //    SpectrumUndefined,
    //};

    double fq_max;
    double detune;
    double M;
    double offset;
    double d;
    // std::optional<BranchEnum> branch = Right;
    std::optional<double> w = std::nullopt;
    std::optional<double> g = std::nullopt;
    // std::optional<SpectrumEnum> spectrum_type = Standard;

    double freq_max;
    double freq_min;

};


double freq2amp_formula(double x, const AcSpectrumParam& params, bool tans2phi);

double amp2freq_formula(double x, const AcSpectrumParam& params, bool tans2phi);

std::vector<double> twoq_pulse(double fwork, double fmax, double tq, int step, const AcSpectrumParam& params);
