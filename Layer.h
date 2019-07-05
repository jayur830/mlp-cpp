#pragma once
#include "Neuron.h"

class Layer {
	int weightScale, _size, act;
	std::vector<Neuron> neurons;
	std::vector<std::vector<double>> inputs, outputs, grad;
public:
	Layer(int, int, int act = SINC);
	size_t size();
	void setActivation(int);
	void forwardProp(std::vector<std::vector<double>>);
	void backProp(std::vector<std::vector<double>>, const double alpha = 0.01);
	std::vector<std::vector<double>>& get();
	std::vector<std::vector<double>> gradient();
	Neuron& operator[](int);
private:
	//std::vector<std::vector<double>> batchNormal(std::vector<std::vector<double>>);
	std::vector<std::vector<double>> transpose(std::vector<std::vector<double>>);
	void softmax();
};