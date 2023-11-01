//
// Created by Daniel on 01.11.2023.
//

#ifndef MLBEGINNER_NEURONNETWORK_H
#define MLBEGINNER_NEURONNETWORK_H

#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#pragma once
using std::vector;
using std::string;
using std::cout;
using std::cin;
using std::endl;
using std::size;
struct neuron {
    double value;
    double error;

    void exponent() {
        value = (1 / (1 + pow(2.71828, -value)));
    }
};
class NeuronNetwork {


    int layers{};
    vector<vector<neuron>> neurons;
    vector<vector<vector<double>>> weights;
    vector<int> size;
public:
    static double SigmoidDerivative(double x);

    double predict(double x);// 0.8 we can change

    NeuronNetwork(int n, int *p, const string& filename);

    void setRandomInput();

    void set_input(vector<double> input); // [0...255]


    void LayersCleaner(int LayerNumber, int start, int stop);

    // Forward Feed helper to calculate
    void ForwardFeeder(int LayerNumber, int start, int stop);

    double ForwardFeed();

    void ErrorCounter(int LayerNumber, int start, int stop, double prediction, double rresult, double lr);

    void WeightsUpdater(int start, int stop, int LayerNum, int lr);

    //Rresult must be any number < 0 if u dont know the right result
    void BackPropogation(double prediction, double rresult,
                         double lr);

    bool SaveWeights();

public:
    NeuronNetwork();

    NeuronNetwork(int n, vector<int> p);
};
#endif //MLBEGINNER_NEURONNETWORK_H
