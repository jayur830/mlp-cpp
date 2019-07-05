#include "FCNN.h"
#include <iostream>
#include <ctime>
#include <numeric>

FCNN::FCNN(std::vector<int> layerNums, const int hiddenAct, const int outputAct) : error(0) {
	for (unsigned i(1); i < layerNums.size() - 1; ++i)
		this->layers.push_back(Layer(layerNums[i - 1], layerNums[i], hiddenAct));
	this->layers.push_back(Layer(layerNums[layerNums.size() - 2], layerNums.back(), this->outputAct = outputAct));
}

void FCNN::forwardProp(std::vector<std::vector<double>> input) {
	this->layers[0].forwardProp(input);
	for (unsigned i(1); i < this->layers.size(); ++i)
		this->layers[i].forwardProp(this->layers[i - 1].get());
	this->output = this->layers.back().get();
}

void FCNN::backProp(const double alpha) {
	this->layers.back().backProp(this->gradient, alpha);
	for (int i(this->layers.size() - 2), j(1); i >= 0; --i, ++j)
		this->layers[i].backProp(this->layers[i + 1].gradient(), alpha);
}

std::vector<std::vector<double>> FCNN::get() {
	return this->output;
}

std::vector<double> FCNN::loss(std::vector<std::vector<double>> target) {
	double total;
	int batchSize(this->output.size()), outputLayerSize(this->output[0].size());
	if (!this->errors.empty()) this->errors.clear();
	if (!this->gradient.empty()) this->gradient.clear();
	for (int i(0); i < batchSize; ++i) {
		total = 0;
		this->gradient.push_back(std::vector<double>(outputLayerSize));
		if (this->outputAct == SOFTMAX) {
			for (int j(0); j < outputLayerSize; ++j) {
				this->gradient[i][j] = this->output[i][j] - target[i][j] + 1e-07;
				if (target[i][j] == 1.0) total = -log(this->output[i][j] + 1e-07);
			}
		}
		else if (this->outputAct == LOGISTIC)
			for (int j(0); j < outputLayerSize; ++j) {
				if (target[i][j] == 1.0) 	total += -log(this->output[i][j] + 1e-07);
				else total += -log(1.0 - this->output[i][j] + 1e-07);
				this->gradient[i][j] = this->output[i][j] - target[i][j] + 1e-07;
			}
		else
			for (int j(0); j < outputLayerSize; ++j) {
				total += (target[i][j] - this->output[i][j]) * (target[i][j] - this->output[i][j]);
				this->gradient[i][j] = this->output[i][j] - target[i][j] + 1e-07;
			}
		this->errors.push_back(
			(total / (this->outputAct == SOFTMAX || this->outputAct == LOGISTIC ? 1.0 : 2.0)));
	}
	return this->errors;
}

double FCNN::loss() {
	std::vector<std::vector<double>> grad(1,
		std::vector<double>(this->gradient[0].size(), 0));
	for (int i(0); i < this->gradient[0].size(); ++i) {
		for (int j(0); j < this->batchSize; ++j)
			grad[0][i] += this->gradient[j][i];
		//grad[0][i] /= this->batchSize;
	}
	this->gradient = grad;
	this->error = std::accumulate(this->errors.begin(), this->errors.end(), this->errors[0]);
	return this->error / this->batchSize;
}

bool FCNN::equal(double a, double b, double epsilon) {
	return abs(a - b) <= epsilon;
}

bool FCNN::equal(std::vector<double> a, std::vector<double> b, double epsilon) {
	for (unsigned i(0); i < a.size(); ++i)
		if (abs(a[i] - b[i]) > epsilon) return false;
	return true;
}

bool FCNN::equal(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b, double epsilon) {
	for (unsigned i(0); i < a.size(); ++i)
		for (unsigned j(0); j < a[i].size(); ++j)
			if (abs(a[i][j] - b[i][j]) > epsilon) return false;
	return true;
}

void FCNN::dataSet(
	std::vector<std::vector<double>> _inputDataSet,
	std::vector<std::vector<double>> _targetDataSet) {
	this->inputDataSet = _inputDataSet;
	this->targetDataSet = _targetDataSet;
	this->batchSize = 1;
	this->iteration = this->inputDataSet.size();
}

void FCNN::setEpoch(int _epoch) {
	this->epoch = _epoch;
}

void FCNN::setBatchSize(int _batchSize) {
	this->batchSize = _batchSize;
	this->iteration = (double)this->inputDataSet.size() / (double)this->batchSize;
}

int FCNN::getEpoch() {
	return this->epoch;
}

int FCNN::getBatchSize() {
	return this->batchSize;
}

void FCNN::train(const double alpha) {
	std::cout << "Total input set size : " << this->inputDataSet.size() << ", Total target set size : " << this->targetDataSet.size() << std::endl;
	clock_t begin, end;
	for (int i(0); i < this->epoch; ++i) {
		std::cout.precision(16);
		std::cout << std::fixed << std::endl << "<" << i + 1 << " epoch>" << std::endl;
		begin = clock();
		double totalError(0);
		for (unsigned j(0), iter(1); j < this->inputDataSet.size(); j += this->batchSize, ++iter) {
			double err;
			while (true) {
				std::cout << "\r[" << iter << " iter] ";
				std::vector<std::vector<double>> inputBatch, targetBatch;
				for (unsigned k(0); k < this->batchSize && j + k < this->inputDataSet.size(); ++k) {
					inputBatch.push_back(this->inputDataSet[j + k]);
					targetBatch.push_back(this->targetDataSet[j + k]);
				}
				forwardProp(inputBatch);
				std::vector<double> error_vector(loss(targetBatch)), compare(error_vector.size(), 0);
				err = loss();
				std::cout << "Cost = " << err << "                    ";
				if (equal(err, 0)) break;
				else backProp(alpha);
			}
			totalError += err;
		}
		end = clock();
		totalError /= this->iteration;
		std::cout << "\r[" << this->iteration << " iter] Cost = " << totalError << ", train time : ";
		std::cout.precision(4);
		std::cout << std::fixed << (double)(end - begin) / 1000.0 << "sec" << std::endl;
	}
	for (unsigned i(0); i < this->layers.size(); ++i)
		for (unsigned j(0); j < this->layers[i].size(); ++j) {
			this->layers[i][j].b_1 = 0.9;
			this->layers[i][j].b_2 = 0.999;
		}
}