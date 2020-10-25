#include "pch.h"
#include <vector>
#include <algorithm>

const size_t n = 5;
const size_t m = 6;
double L = 0.33;
double R = 0.07;
double dx = L / (m - 1);
double dy = 2 * R / (n - 1);
double mu = 1.0019e-3;
double ro = 1e3;
double yx = dy / dx;
double xy = dx / dy;

double u_aE(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double u_aEl(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double u_aW(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double u_aWr(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double u_aN(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double u_aS(std::vector<std::vector<double>> &matrix, size_t i, size_t j);

double v_aE(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double v_aEl(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double v_aW(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double v_aWr(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double v_aN(std::vector<std::vector<double>> &matrix, size_t i, size_t j);
double v_aS(std::vector<std::vector<double>> &matrix, size_t i, size_t j);

void Copy(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B, size_t row);

int main() 
{
	double p2 = 1e5;
	double p1 = p2 + 2.897e-4;
	int maxiteration = 300;

	std::vector<std::vector<double>> p_zv(n);
	std::vector<std::vector<double>> p_p(n);
	std::vector<std::vector<double>> p(n);
	for (size_t i = 0; i < n; i++) {
		p_zv[i].resize(m);
		p_p[i].resize(m);
		p[i].resize(m);
	}

	std::vector<std::vector<double>> uOld(n);
	std::vector<std::vector<double>> u_zv(n);
	std::vector<std::vector<double>> utemp(n);
	std::vector<std::vector<double>> uAP(n);
	for (size_t i = 0; i < n; i++) {
		uOld[i].resize(m - 1);
		u_zv[i].resize(m - 1);
		utemp[i].resize(m - 1);
		uAP[i].resize(m - 1);
	}

	std::vector<std::vector<double>> U(n*(m - 1));
	for (size_t i = 0; i < n*(m - 1); i++)
		U[i].resize(n*(m - 1));
	std::vector<double> bu(n*(m - 1));

	std::vector<std::vector<double>> vOld(n + 1);
	std::vector<std::vector<double>> v_zv(n + 1);
	std::vector<std::vector<double>> vtemp(n + 1);
	std::vector<std::vector<double>> vAP(n + 1);
	for (size_t i = 0; i < n + 1; i++) {
		vOld[i].resize(m);
		v_zv[i].resize(m);
		vtemp[i].resize(m);
		vAP[i].resize(m);
	}

	std::vector<std::vector<double>> V((n + 1)*m);
	for (size_t i = 0; i < (n + 1)*m; i++)
		V[i].resize((n + 1)*m);
	std::vector<double> bv((n + 1)*m);

	int iteration = 0;
	while (iteration < maxiteration)
	{
		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < m - 1; j++) {
				if (i == 0 || i == n - 1) {
					utemp[i][j] = 1.;
					uAP[i][j] = u_aS(vOld, i, j) + u_aN(vOld, i, j) + 2.*mu*yx;
					bu[i*(m - 1) + j] = 0.;
					Copy(U, utemp, i*(m - 1) + j);
					Fill(utemp, 0.);
					continue;
				}

				if (i > 0 && i < n - 1 && j == 0) {
					utemp[i][j] = u_aEl(uOld, i, j) + mu * 3. * yx + u_aN(vOld, i, j) + u_aS(vOld, i, j);
					uAP[i][j] = utemp[i][j];
					utemp[i - 1][j] = -u_aN(vOld, i, j);
					utemp[i][j + 1] = -u_aEl(uOld, i, j);
					utemp[i + 1][j] = -u_aS(vOld, i, j);
					bu[i*(m - 1) + j] = (p_zv[i][j] - p_zv[i][j + 1])*dy;
					Copy(U, utemp, i*(m - 1) + j);
					Fill(utemp, 0.);
					continue;
				}

				if (i > 0 && i < n - 1 && j == m - 2) {
					utemp[i][j] = u_aWr(uOld, i, j) + mu * 3. * yx + u_aN(vOld, i, j) + u_aS(vOld, i, j);
					uAP[i][j] = utemp[i][j];
					utemp[i][j - 1] = -u_aWr(uOld, i, j);
					utemp[i - 1][j] = -u_aN(vOld, i, j);
					utemp[i + 1][j] = -u_aS(vOld, i, j);
					bu[i*(m - 1) + j] = (p_zv[i][j] - p_zv[i][j + 1])*dy;
					Copy(U, utemp, i*(m - 1) + j);
					Fill(utemp, 0.);
					continue;
				}

				else
				{
					utemp[i][j] = u_aE(uOld, i, j) + u_aW(uOld, i, j) + u_aN(vOld, i, j) + u_aS(vOld, i, j);
					uAP[i][j] = utemp[i][j];
					utemp[i][j - 1] = -u_aW(uOld, i, j);
					utemp[i][j + 1] = -u_aE(uOld, i, j);
					utemp[i - 1][j] = -u_aN(vOld, i, j);
					utemp[i + 1][j] = -u_aS(vOld, i, j);
					bu[i*(m - 1) + j] = (p_zv[i][j] - p_zv[i][j + 1])*dy;
					Copy(U, utemp, i*(m - 1) + j);
					Fill(utemp, 0.);
					continue;
				}
			}
		}

		for (size_t i = 0; i < n + 1; i++) {
			for (size_t j = 0; j < m; j++) {
				if (i == 0 || i == n) {
					vtemp[i][j] = 1.;
					vAP[i][j] = 2. * mu * yx + 2. * mu * xy;
					bv[i*m + j] = 0.;
					Copy(V, vtemp, i*m + j);
					Fill(vtemp, 0.);
					continue;
				}

				if (i > 0 && i < n && j == 0) {
					vtemp[i][j] = v_aEl(uOld, i, j) + mu * 4. * yx + v_aN(vOld, i, j) + v_aS(vOld, i, j);
					vAP[i][j] = vtemp[i][j];
					vtemp[i - 1][j] = -v_aN(vOld, i, j);
					vtemp[i][j + 1] = -v_aEl(uOld, i, j);
					vtemp[i + 1][j] = -v_aS(vOld, i, j);
					bv[i*m + j] = (p_zv[i][j] - p_zv[i - 1][j])*dy;
					Copy(V, vtemp, i*m + j);
					Fill(vtemp, 0.);
					continue;
				}

				if (i > 0 && i < n && j == m - 1) {
					vtemp[i][j] = v_aWr(uOld, i, j) + 4.*mu*yx + v_aN(vOld, i, j) + v_aS(vOld, i, j);
					vAP[i][j] = vtemp[i][j];
					vtemp[i][j - 1] = -v_aWr(uOld, i, j);
					vtemp[i - 1][j] = -v_aN(vOld, i, j);
					vtemp[i + 1][j] = -v_aS(vOld, i, j);
					bv[i*m + j] = (p_zv[i][j] - p_zv[i - 1][j])*dy;
					Copy(V, vtemp, i*m + j);
					Fill(vtemp, 0.);
					continue;
				}

				else {
					vtemp[i][j] = v_aE(uOld, i, j) + v_aW(uOld, i, j) + v_aN(vOld, i, j) + v_aS(vOld, i, j);
					vAP[i][j] = vtemp[i][j];
					vtemp[i][j - 1] = -v_aW(uOld, i, j);
					vtemp[i][j + 1] = -v_aE(uOld, i, j);
					vtemp[i - 1][j] = -v_aN(vOld, i, j);
					vtemp[i + 1][j] = -v_aS(vOld, i, j);
					bv[i*m + j] = (p_zv[i][j] - p_zv[i - 1][j])*dy;
					Copy(V, vtemp, i*m + j);
					Fill(vtemp, 0.);
					continue;
				}
			}
		}


	}

	return 0;
}

void SORMethod(std::vector<std::vector<double>> &mass, std::vector<double> &b,
	std::vector<double> &x_0, std::vector<double> &x_2, double w) {
	std::vector<double> x(b.size());
	std::vector<double> x_1(b.size());
	double elemA = 0;
	x_1 = x_0;
	int iter = 0;
	double delta = 0;
	double sum1 = 0;
	double sum2 = 0;
	double epsilon = 1.0e-4;

	std::vector<double> A(mass.size()*b.size());
	for(int i = 0; i < )

	do {
		iter++;
		if (iter > 100) {
			x_2 = x;
			break;
		}
		for (int i = 0; i < b.size(); i++) {
			sum1 = 0;
			sum2 = 0;
			for (int j = 0; i < i; j++) {
				if (abs(i - j) > mass.size())
					elemA = 0;
				else
					elemA = A[i + (j - i + A.size())*b.size()];
			}
		}
	} while (iter <= 100 && delta > epsilon);
}

void Fill(std::vector<std::vector<double>> &A, double value) {
	for (size_t i = 0; i < A.size(); i++) {
		for (size_t j = 0; j < A[i].size(); j++)
			A[i][j] = value;
	}
}

void Copy(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B, size_t row) {
	for (size_t i = 0; i < B.size(); i++) {
		for (size_t j = 0; j < B[i].size(); j++) {
			A[row][i * B[i].size() + j] = B[i][j];
		}
	}
}

double u_aE(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double De = mu * yx;
	double Fe = ro * dy * (matrix[i][j] + matrix[i][j + 1]) / 2.;
	return De + std::max(-Fe, 0.);
}

double u_aEl(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double De = mu * yx;
	double De2 = mu * yx / 3.;
	double Fe = ro * dy * (matrix[i][j] + matrix[i][j + 1]) / 2.;
	return De + std::max(-Fe, 0.) + De2;
}

double u_aW(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Dw = mu * yx;
	double Fw = ro * dy * (matrix[i][j] + matrix[i][j - 1]) / 2.;
	return Dw + std::max(Fw, 0.);
}

double u_aWr(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Dw = mu * yx;
	double Dw2 = mu * yx / 3.;
	double Fw = ro * dy * (matrix[i][j] + matrix[i][j - 1]) / 2.;
	return Dw + std::max(Fw, 0.) + Dw2;
}

double u_aN(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Dn = mu * xy;
	double Fn = ro * dx * (matrix[i][j] + matrix[i][j + 1]) / 2.;
	return Dn + std::max(-Fn, 0.);
}

double u_aS(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Ds = mu * xy;
	double Fs = ro * dx * (matrix[i + 1][j] + matrix[i + 1][j + 1]) / 2.;
	return Ds + std::max(Fs, 0.);
}

double v_aE(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double De = mu * yx;
	double Fe = ro * dy * (matrix[i - 1][j] + matrix[i][j]) / 2.;
	return De + std::max(-Fe, 0.);
}

double v_aEl(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double De = mu * yx;
	double De2 = mu * yx;
	double Fe = ro * dy * (matrix[i - 1][j] + matrix[i][j]) / 2.;
	return De + std::max(-Fe, 0.) + De2;
}

double v_aW(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Dw = mu * yx;
	double Fw = ro * dy * (matrix[i - 1][j - 1] + matrix[i][j - 1]) / 2.;
	return Dw + std::max(Fw, 0.);
}

double v_aWr(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Dw = mu * yx;
	double Dw2 = mu * yx;
	double Fw = ro * dy * (matrix[i - 1][j - 1] + matrix[i][j - 1]) / 2.;
	return Dw + std::max(Fw, 0.) + Dw2;
}

double v_aN(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Dn = mu * xy;
	double Fn = ro * dx * (matrix[i][j] + matrix[i - 1][j]) / 2.;
	return Dn + std::max(-Fn, 0.);
}

double v_aS(std::vector<std::vector<double>> &matrix, size_t i, size_t j) {
	double Ds = mu * xy;
	double Fs = ro * dx * (matrix[i][j] + matrix[i + 1][j]) / 2.;
	return Ds + std::max(Fs, 0.);
}