#include "Layer.h"
#include <iostream>
#include <numeric>

Layer::Layer(int _weightScale, int neuronNum, int act) :
	weightScale(_weightScale), _size(neuronNum), act(act) {
	this->grad = std::vector<std::vector<double>>(1, std::vector<double>(this->weightScale, 0));
	for (int i(0); i < this->_size; ++i)
		this->neurons.push_back(Neuron(this->weightScale, act));
}

size_t Layer::size() {
	return this->_size;
}

void Layer::setActivation(int act) {
	for (int i(0); i < this->_size; ++i)
		this->neurons[i].setActivation(act);
}

void Layer::forwardProp(std::vector<std::vector<double>> _input) {
	this->inputs = _input;
	if (!this->outputs.empty()) this->outputs.clear();
	for (int i(0); i < this->_size; ++i) {
		this->neurons[i].forwardProp(this->inputs);
		this->outputs.push_back(this->neurons[i].get());
	}
	this->outputs = transpose(this->outputs);
	if (this->act == SOFTMAX) softmax();
}

void Layer::backProp(std::vector<std::vector<double>> grad, const double alpha) {
	int batchSize(grad.size());
	this->grad = std::vector<std::vector<double>>(
		batchSize, std::vector<double>(this->weightScale, 0));
	std::vector<std::vector<double>> g;
	grad = transpose(grad);
	for (int i(0); i < this->_size; ++i) {
		this->neurons[i].backProp(grad[i], alpha);
		g = this->neurons[i].gradient();
		for (int j(0); j < batchSize; ++j)
			for (int k(0); k < this->weightScale; ++k)
				this->grad[j][k] += g[j][k];
	}
}

std::vector<std::vector<double>>& Layer::get() {
	return this->outputs;
}

std::vector<std::vector<double>> Layer::gradient() {
	return this->grad;
}

Neuron& Layer::operator[](int index) {
	return this->neurons[index];
}

//std::vector<std::vector<double>> Layer::batchNormal(
//	std::vector<std::vector<double>> miniBatch) {
//	int layerSize(miniBatch.size()), batchSize(miniBatch[0].size());
//	std::vector<double> mean, variance;
//	for (int i(0); i < layerSize; ++i) {
//		mean.push_back(0);
//		variance.push_back(0);
//		for (int j(0); j < batchSize; ++j)
//			mean[i] += miniBatch[i][j];
//		mean[i] /= batchSize;
//		for (int j(0); j < batchSize; ++j)
//			variance[i] += (miniBatch[i][j] - mean[j]) * (miniBatch[i][j] - mean[j]);
//		variance[i] /= batchSize;
//	}
//	return mini
//}

std::vector<std::vector<double>> Layer::transpose(
	std::vector<std::vector<double>> matrix) {
	std::vector<std::vector<double>>
		trans(matrix[0].size(), std::vector<double>(matrix.size()));
	for (unsigned i(0); i < trans.size(); ++i)
		for (unsigned j(0); j < trans[i].size(); ++j)
			trans[i][j] = matrix[j][i];
	return trans;
}

void Layer::softmax() {
	double total(0), max(0);
	std::vector<std::vector<double>> output(this->outputs);
	this->outputs.clear();
	this->outputs = std::vector<std::vector<double>>(output.size());
	for (unsigned i(0); i < output.size(); ++i) {
		//for (double out : output[i]) std::cout << out << " ";
		//std::cout << std::endl;
		total = 0; max = 0;
		for (double out : output[i]) max = max > out ? max : out;
		for (double& out : output[i]) total += (out = exp(out - max));
		//std::cout << i << " total = " << total << std::endl;
		//double _total(0);
		//std::cout << "[";
		for (double out : output[i]) {
			this->outputs[i].push_back(out / total);
		//	if (out / total <= 1.0) std::cout << "O, ";
		//	else std::cout << "X, ";
		}
		//std::cout << "\b\b]\n";
	}
}