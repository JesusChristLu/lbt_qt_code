#include "interpolation.h"

Interp1d::Interp1d(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must be the same size");
    }
    if (x.size() < 2) {
        throw std::invalid_argument("Size of x and y must be at least 2");
    }
    for (int i = 0; i < x.size(); ++i)
    {
        x_y.emplace_back(x[i], y[i]);
    }
    std::sort(x_y.begin(), x_y.end(),
        [](const std::pair<double, double>& v1, const std::pair<double, double>& v2)
        {
            return v1.first < v2.first;
        }
    );
}

double Interp1d::operator()(double xp) const {
    // Find the interval in x that contains xp

    /*if (x_y.size() == 2)
    {
        auto&& [x1, y1] = x_y[0];
        auto&& [x2, y2] = x_y[1];
        double t = (xp - x1) / (x2 - x1);
        return y1 * (1 - t) + y2 * t;
    }*/

    auto it = std::lower_bound(x_y.begin(), x_y.end(), xp,
        [](const std::pair<double, double>& x_y, double v)
        {
            return x_y.first < v;
        });
    if (it == x_y.end()) {
        // it = x_y.begin() + x_y.size() - 2;
        throw std::out_of_range("Interpolation point is out of range");
    }
    else if (it == x_y.begin()) {
        // it++;
        throw std::out_of_range("Interpolation point is out of range");
    }

    // Calculate the interpolation
    int idx = it - x_y.begin() - 1;
    auto&& [x1, y1] = x_y[idx];
    auto&& [x2, y2] = x_y[idx + 1];

    double t = (xp - x1) / (x2 - x1);
    return y1 * (1 - t) + y2 * t;
}

CubicSpline::CubicSpline(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must be the same size");
    }
    if (x.size() < 3) {
        throw std::invalid_argument("Size of x and y must be at least 3 for cubic spline interpolation");
    }

    int n = x.size() - 1;
    std::vector<double> a(y.begin(), y.end());
    std::vector<double> b(n), d(n), h(n), alpha(n), c(n + 1), l(n + 1), mu(n + 1), z(n + 1);

    for (int i = 0; i < n; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    alpha[0] = 0; // Initialized to 0 for natural spline
    for (int i = 1; i < n; ++i) {
        alpha[i] = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1]);
    }

    l[0] = 1;
    mu[0] = z[0] = 0;

    for (int i = 1; i < n; ++i) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n] = 1;
    z[n] = c[n] = 0;

    for (int j = n - 1; j >= 0; --j) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    for (int i = 0; i < n; ++i) {
        splines_.push_back({ a[i], b[i], c[i], d[i], x[i] });
    }
    std::sort(splines_.begin(), splines_.end(), 
        [](const SplineTuple& a, const SplineTuple& b) {return a.x < b.x; }
    );
}

double CubicSpline::operator()(double xp) const {
    if (splines_.size() == 0)
        ThrowRuntimeError("Bad interpolation object (input is empty).");
    // Find the right place in the table
    if (xp <= splines_[0].x) {
        return splines_[0](xp);
    }

    if (xp >= splines_.back().x) {
        return splines_.back()(xp);
    }

    auto it = std::lower_bound(splines_.begin(), splines_.end(), xp,
        [](const SplineTuple& s, double xp) { return s.x < xp; });
    --it; // Correct position

    return (*it)(xp);
}

LinearInterpolation2D::LinearInterpolation2D(const std::vector<double>& x, const std::vector<double>& y, const std::vector<std::vector<double>>& z)
    : x_(x), y_(y), z_(z) {
    if (x.empty() || y.empty() || z.empty() || z[0].empty()) {
        throw std::runtime_error("Input vectors must not be empty.");
    }
    if (z.size() != x.size() || z[0].size() != y.size()) {
        throw std::runtime_error("Dimensions of z must match lengths of x and y.");
    }
}

double LinearInterpolation2D::operator()(double x, double y) const {
    // Find the indices of the grid cells that x and y fall into
    int i = findInterval(x_, x);
    int j = findInterval(y_, y);

    // Perform bilinear interpolation
    double x1 = x_[i], x2 = x_[i + 1];
    double y1 = y_[j], y2 = y_[j + 1];
    double z11 = z_[i][j], z12 = z_[i][j + 1], z21 = z_[i + 1][j], z22 = z_[i + 1][j + 1];

    double r1 = ((x2 - x) / (x2 - x1)) * z11 + ((x - x1) / (x2 - x1)) * z21;
    double r2 = ((x2 - x) / (x2 - x1)) * z12 + ((x - x1) / (x2 - x1)) * z22;

    return ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2;

}

//static double linearInterpolate(double x, double x0, double x1, double y0, double y1) {
//    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
//}

int LinearInterpolation2D::findInterval(const std::vector<double>& v, double value) {
    if (value < v.front() || value > v.back()) {
        throw std::runtime_error("Interpolation value out of range.");
    }
    /*if (value <= v.front()) return 0;
    if (value >= v.back()) return v.size() - 2;*/

    auto it = std::lower_bound(v.begin(), v.end(), value);
    if (it != v.end() && it != v.begin()) {
        return it - v.begin() - 1;
    }
    throw std::runtime_error("Interval not found.");
}

CubicInterpolation2D::CubicInterpolation2D(const std::vector<double>& x, const std::vector<double>& y, const std::vector<std::vector<double>>& z)
    : x_(x), y_(y), z_(z) {
    if (y_.size() != z_.size()) {
        throw std::runtime_error("Row count of z must match size of y.");
    }
    for (const auto& row : z_) {
        if (x_.size() != row.size()) {
            throw std::runtime_error("Column count of each row in z must match size of x.");
        }
    }

    // Create cubic splines for each row
    for (const auto& row : z_) {
        rowSplines_.push_back(CubicSpline(x, row));
    }

    // Create cubic splines for each column
    for (size_t i = 0; i < x_.size(); ++i) {
        std::vector<double> col(z_.size());
        for (size_t j = 0; j < z_.size(); ++j) {
            col[j] = z_[j][i];
        }
        colSplines_.push_back(CubicSpline(y, col));
    }
}

double CubicInterpolation2D::operator()(double x, double y) const
{
    // Interpolate each row spline at x
    std::vector<double> rowValues;
    for (const auto& spline : rowSplines_) {
        rowValues.push_back(spline(x));
    }

    // Interpolate these values along y
    CubicSpline colSpline(y_, rowValues);
    return colSpline(y);
}
