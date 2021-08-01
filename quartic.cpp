#include <iostream>
#include <array>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <bits/stdc++.h>
#include <boost/math/special_functions/bessel.hpp>
#include <iomanip>
#include <getopt.h>
#include "modules/array.cpp"

using namespace std;
using namespace boost::numeric::odeint;

using state_type = array<double, 4>;

#define no_argument 0
#define required_argument 1
#define optional_argument 2

const double t_init = 1e-9;

double t_fin = 1.0e10;
double V0 = 1.e2;
double lambda = 5.e-6;
double phi_i = 1.e-3;
double dphi_i = 0.;

string float_to_string(float f, int digits)
{
    std::ostringstream oss;
    oss << scientific << setprecision(digits) << f;
    return oss.str();
}

double V(double phi)
{
    return V0 * (1 - lambda * pow(phi, 4));
}

double Vphi(double phi)
{
    return -4. * V0 * lambda * pow(phi, 3);
}

double Vphiphi(double phi)
{
    return -12. * V0 * lambda * pow(phi, 2);
}

double Hubble(double phi, double dphi)
{
    return sqrt(1. / 3. * (0.5 * pow(dphi, 2) + V(phi)));
}

double epsilon(double phi)
{
    return 0.5 * pow(Vphi(phi) / V(phi), 2);
}

double eta(double phi)
{
    return Vphiphi(phi) / V(phi);
}

void solve_BG(const state_type &y, state_type &dy, const double t)
{
    dy[0] = y[1];
    dy[1] = -3. * y[3] * y[1] - Vphi(y[0]);
    dy[2] = y[3];
    dy[3] = -1. / 2. * pow(y[1], 2);
}

int main(int argc, char *argv[])
{
    int opt;
    int longindex = 0;
    const struct option longopts[] = {
        {"V0", optional_argument, NULL, 'v'},
        {"lambda", optional_argument, NULL, 'l'},
        {"phi", optional_argument, NULL, 'p'},
        {"t_fin", optional_argument, NULL, 't'},
        {0, 0, 0, 0},
    };
    while ((opt = getopt_long(argc, argv, "v:l:p:t:", longopts, &longindex)) != -1)
    {
        switch (opt)
        {
        case 'v':
            V0 = stod(optarg);
            break;
        case 'l':
            lambda = stod(optarg);
            break;
        case 'p':
            phi_i = stod(optarg);
            break;
        case 't':
            t_fin = stod(optarg);
            break;
        }
    }

    const char *fileName = "test.dat";
    std::ofstream ofs(fileName);
    double Dphi_i = -Vphi(phi_i) / sqrt(3. * V(phi_i));
    //初期状態
    ofs << "#t phi dphi N H epsilon eta H_n/H" << endl;
    state_type state0 = {phi_i, Dphi_i, 0., Hubble(phi_i, Dphi_i)};

    auto Stepper = make_dense_output<runge_kutta_dopri5<state_type>>(1.0e-20, 1.0e-10);

    int v_param = 0;
    vector<double> v = logspace(log10(t_init), log10(t_fin), 100000);

    cout << "phimax:" << 1. / pow(lambda, 0.25) << endl;
    if (phi_i > 1. / pow(lambda, 0.25))
    {
        exit(1);
    }

    //数値解析を実行
    //ステップごとにステップ幅を変えない
    integrate_adaptive(Stepper,                                 //ステップごとの手法
                       solve_BG,                                //状態方程式
                       state0,                                  //初期値
                       t_init,                                  //初期時刻
                       t_fin,                                   //終了時刻
                       0.0001,                                  //ステップ幅
                       [&](const state_type &x, const double t) //ステップ毎に実行される関数.
                       {
                           if (v[v_param] <= t)
                           {
                               v_param++;
                               ofs << scientific << setprecision(14) << t << "," << x[0] << "," << x[1] << "," << x[2] << "," << x[3] << "," << epsilon(x[0]) << "," << eta(x[0]) << "," << abs((Hubble(x[0], x[1]) - x[3]) / x[3]) << endl;
                           }
                           if (epsilon(x[0]) >= 1.)
                           {
                               cout << "N:" << x[2] << "\nepsilon:" << epsilon(x[0]) << endl;
                               exit(0);
                           }
                           if (abs((Hubble(x[0], x[1]) - x[3]) / x[3]) > 0.2)
                           {
                               {
                                   ofs << "Large error" << endl;
                                   cout << "N:" << x[2] << "\nepsilon:" << epsilon(x[0]) << "\nLarge error" << endl;
                                   exit(1);
                               }
                           }
                       });
}
