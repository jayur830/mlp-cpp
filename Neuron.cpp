#include "Neuron.h"
#include <iostream>
#define EPSILON pow(10, -8)
#define BETA_1 0.9
#define BETA_2 0.999
#define A 0.05
#define ALPHA 1.67326
#define LAMBDA 1.0507
#define P -0.1

Neuron::Neuron(int _weightScale, const int _act) :
	weightScale(_weightScale), bias(0.0001), act(_act) {
	for (int i(0); i < this->weightScale; ++i) {
		this->weights.push_back(((double)rand() / (double)RAND_MAX) * 0.01);
		this->_grad.push_back(std::vector<double>());
	}
	this->_grad.push_back(std::vector<double>());
	this->actFunc[IDENTITY] = [](double x) -> double { return x; };
	this->actFunc[LOGISTIC] = [](double x) -> double { return 1.0 / (1.0 + exp(-x)); };
	this->actFunc[TANH] = [](double x) -> double { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); };
	this->actFunc[ARCTAN] = [](double x) -> double { return atan(x); };
	this->actFunc[ARSINH] = [](double x) -> double { return log(x + sqrt((x * x) + 1)); };
	this->actFunc[SOFTSIGN] = [](double x) -> double { return x / (1.0 + abs(x)); };
	this->actFunc[ISRU] = [](double x) -> double { return x / sqrt(1.0 + (ALPHA * x * x)); };
	this->actFunc[ISRLU] = [](double x) -> double { return x >= 0 ? x : x / sqrt(1.0 + (ALPHA * x * x)); };
	this->actFunc[SQNL] = [](double x) -> double {
		if (x > 2.0) return 1.0;
		else if (x >= 0 && x <= 2.0) return x - ((x * x) / 4.0);
		else if (x >= -2.0 && x < 0) return x + ((x * x) / 4.0);
		else return -1.0;
	};
	this->actFunc[RELU] = [](double x) -> double { return x > 0 ? x : 0; };
	this->actFunc[LRELU] = [](double x) -> double { return x >= 0 ? x : x * 0.01; };
	this->actFunc[PRELU] = [](double x) -> double { return x >= 0 ? x : x * A; };
	this->actFunc[ELU] = [](double x) -> double { return x > 0 ? x : ALPHA * (exp(x) - 1); };
	this->actFunc[SELU] = [](double x) -> double { return x > 0 ? LAMBDA * x : LAMBDA * ALPHA * (exp(x) - 1); };
	this->actFunc[SOFTPLUS] = [](double x) -> double { return log(1 + exp(x)); };
	this->actFunc[BESTID] = [](double x) -> double { return ((sqrt((x * x) + 1.0) - 1.0) / 2.0) + x; };
	this->actFunc[SOFTEXP] = [](double x) -> double { return ALPHA > 0 ? ((exp(ALPHA * x) - 1) / ALPHA) + ALPHA : (ALPHA < 0 ? -(log(1 - (ALPHA * (x + ALPHA))) / ALPHA) : x); };
	this->actFunc[SINUSOID] = [](double x) -> double { return sin(x); };
	this->actFunc[SINC] = [](double x) -> double { return x == 0 ? 1.0 : sin(x) / x; };
	this->actFunc[GAUSSIAN] = [](double x) -> double { return exp(-x * x); };
	this->actFunc[SOFTMAX] = [](double x) -> double { return x; };

	this->grad[IDENTITY] = [](double x) -> double { return 1.0; };
	this->grad[LOGISTIC] = [](double x) -> double { return (x = 1.0 / (1.0 + exp(-x))) * (1.0 - x); };
	this->grad[TANH] = [](double x) -> double { return 1.0 - ((x = (exp(x) - exp(-x)) / (exp(x) + exp(-x))) * x); };
	this->grad[ARCTAN] = [](double x) -> double { return 1.0 / ((x * x) + 1.0); };
	this->grad[ARSINH] = [](double x) -> double { return 1.0 / sqrt((x * x) + 1.0); };
	this->grad[SOFTSIGN] = [](double x) -> double { return 1.0 / ((1.0 + abs(x)) * (1.0 + abs(x))); };
	this->grad[ISRU] = [](double x) -> double {	return (x = 1.0 / sqrt(1.0 + (ALPHA * x * x))) * x * x; };
	this->grad[ISRLU] = [](double x) -> double { return x >= 0 ? 1.0 : (x = 1.0 / sqrt(1.0 + (ALPHA * x * x))) * x * x; };
	this->grad[RELU] = [](double x) -> double { return x > 0 ? 1.0 : 0; };
	this->grad[LRELU] = [](double x) -> double { return x >= 0 ? 1 : 0.01; };
	this->grad[PRELU] = [](double x) -> double { return x >= 0 ? 1.0 : A; };
	this->grad[ELU] = [](double x) -> double { return x > 0 ? 1 : ALPHA * exp(x); };
	this->grad[SELU] = [](double x) -> double { return x > 0 ? LAMBDA : LAMBDA * ALPHA * exp(x); };
	this->grad[SOFTPLUS] = [](double x) -> double { return 1 / (1 + exp(-x)); };
	this->grad[BESTID] = [](double x) -> double { return (x / (2.0 * sqrt((x * x) + 1.0))) + 1.0; };
	this->grad[SOFTEXP] = [](double x) -> double { return ALPHA >= 0 ? exp(ALPHA * x) : 1 / (1 - (ALPHA * (ALPHA * x))); };
	this->grad[SINUSOID] = [](double x) -> double { return cos(x); };
	this->grad[SINC] = [](double x) -> double { return x == 0 ? 0 : (cos(x) - x) - (sin(x) / (x * x)); };
	this->grad[GAUSSIAN] = [](double x) -> double { return -2.0 * x * exp(-x * x); };
	this->grad[SOFTMAX] = [](double x) -> double { return 1.0; };
}

void Neuron::setActivation(const int _act) {
	this->act = _act;
}

size_t Neuron::size() {
	return this->weightScale;
}

void Neuron::forwardProp(std::vector<std::vector<double>> x) {
	this->inputs = x;
	int batchSize(this->inputs.size());
	if (!this->sigma.empty()) this->sigma.clear();
	for (int i(0); i < batchSize; ++i) {
		this->sigma.push_back(this->bias);
		for (int j(0); j < this->weightScale; ++j)
			this->sigma[i] += this->weights[j] * this->inputs[i][j];
	}
	//std::cout << this->inputs[0][0] << std::endl;
	this->activation = activate();
}

void Neuron::backProp(std::vector<double> grad, const double alpha) {
	//double df_ds(this->grad[this->act](this->sigma)), dE_ds(grad * df_ds);
	//for (int i(0); i < this->weightScale; ++i) {
	//	this->_grad[i] = dE_ds * this->weights[i];
	//	this->v[i] *= 0.9;
	//	this->v[i] -= alpha * dE_ds * this->inputs[i];
	//	this->weights[i] += this->v[i];
	//}
	//this->_grad.back() = dE_ds * this->weights.back();
	//this->v.back() *= 0.9;
	//this->v.back() -= alpha * dE_ds;
	//this->bias += this->v.back();

	int batchSize(grad.size());
	if (!this->_grad.empty()) this->_grad.clear();
	if (this->v.size() != batchSize) {
		this->v = std::vector<std::vector<double>>(batchSize, std::vector<double>(this->weightScale + 1, 0));
		this->g = std::vector<std::vector<double>>(batchSize, std::vector<double>(this->weightScale + 1, 0));
		this->_v = std::vector<std::vector<double>>(batchSize, std::vector<double>(this->weightScale + 1, 0));
		this->_g = std::vector<std::vector<double>>(batchSize, std::vector<double>(this->weightScale + 1, 0));
	}
	for (int i(0); i < batchSize; ++i) {
		double df_ds(this->act == SOFTMAX ? 1.0 : this->grad[this->act](this->sigma[i])), dE_ds(grad[i] * df_ds);
		this->_grad.push_back(std::vector<double>(this->weightScale + 1));
		for (int j(0); j < this->weightScale; ++j) {
			adam(dE_ds, this->inputs[i][j], i, j);
			this->weights[j] -= this->_v[i][j] * alpha / (sqrt(this->_g[i][j]) + EPSILON);
		}
		adam(dE_ds, 1.0, i, this->weightScale);
		this->bias -= this->_v[i][this->_v[i].size() - 1] * alpha / (sqrt(this->_g[i][this->_g[i].size() - 1]) + EPSILON);
		this->b_1 *= BETA_1;
		this->b_2 *= BETA_2;
	}
}

std::vector<double>& Neuron::get() {
	return this->activation;
}

std::vector<std::vector<double>> Neuron::gradient() {
	return this->_grad;
}

void Neuron::adam(double gradient, double input, int i, int j) {
	if (j != this->weightScale) this->_grad[i][j] = gradient * this->weights[j];
	else this->_grad[i][j] = gradient;
	this->v[i][j] = (this->v[i][j] * BETA_1) + ((1 - BETA_1) * gradient * input);
	this->g[i][j] = (this->g[i][j] * BETA_2) + ((1 - BETA_2) * gradient * input * gradient * input);
	this->_v[i][j] = this->v[i][j] / (1 - this->b_1);
	this->_g[i][j] = this->g[i][j] / (1 - this->b_2);
}

void Neuron::weightUpdate() {
	for (int i(0); i < this->weightScale; ++i) {
	}
}

std::vector<double> Neuron::activate() {
	std::vector<double> act;
	for (unsigned i(0); i < this->sigma.size(); ++i)
		act.push_back(this->actFunc[this->act](this->sigma[i]));
	return act;
}