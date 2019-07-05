#pragma once
#include <vector>
enum Activation { 
	IDENTITY,
	LOGISTIC,
	TANH,
	ARCTAN,
	ARSINH,
	SOFTSIGN,
	ISRU,
	ISRLU,
	SQNL,
	RELU,
	LRELU,
	PRELU,
	ELU,
	SELU,
	SOFTPLUS,
	BESTID,
	SOFTEXP,
	SINUSOID,
	SINC,
	GAUSSIAN,
	SOFTMAX,
	SIZE };

class Neuron {
	friend class FCNN;
	int weightScale, act;
	std::vector<std::vector<double>> inputs, _grad, v, g, _v, _g;
	std::vector<double> weights, sigma, activation;
	double bias, (*actFunc[SIZE])(double), (*grad[SIZE])(double), b_1 = 0.9, b_2 = 0.999;
public:
	Neuron(int, const int _act = SINC);
	void setActivation(const int);
	size_t size();
	void forwardProp(std::vector<std::vector<double>>);
	void backProp(std::vector<double>, const double alpha = 0.01);
	std::vector<double>& get();
	std::vector<std::vector<double>> gradient();
private:
	void adam(double, double, int, int);
	void weightUpdate();
	std::vector<double> activate();
};