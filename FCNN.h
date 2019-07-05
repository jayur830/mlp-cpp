#pragma once
#include "Layer.h"

class FCNN {
	std::vector<Layer> layers;
	std::vector<double> errors;
	std::vector<std::vector<double>> output, gradient;
	std::vector<std::vector<double>> inputDataSet, targetDataSet;
	int epoch = 100, batchSize, iteration, outputAct;
	double error;
public:
	FCNN(std::vector<int>, const int, const int outputAct = SINC);
	void forwardProp(std::vector<std::vector<double>>);
	void backProp(const double alpha = 0.01);
	std::vector<std::vector<double>> get();
	std::vector<double> loss(std::vector<std::vector<double>>);
	double loss();

	bool equal(double, double, double epsilon = 0.1);
	bool equal(std::vector<double>, std::vector<double>, double epsilon = 0.1);
	bool equal(std::vector<std::vector<double>>, std::vector<std::vector<double>>, double epsilon = 0.1);

	void dataSet(
		std::vector<std::vector<double>>, 
		std::vector<std::vector<double>>);
	void setEpoch(int);
	void setBatchSize(int);
	int getEpoch();
	int getBatchSize();
	void train(const double alpha = 0.01);
};