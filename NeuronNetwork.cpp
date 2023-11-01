#include "NeuronNetwork.h"
#include <random>

double NeuronNetwork::SigmoidDerivative(double x) {
    if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
    double res = x * (1.0 - x);
    return res;
}

double NeuronNetwork::predict(double x) {// 0.8 we can change
    if (x >= 0.8) {
        return 1;
    } else {
        return 0;
    }
}

NeuronNetwork::NeuronNetwork(int n, int *p, const string &filename) {
    std::ifstream fin;
    fin.open(filename);
    layers = n;
    neurons = vector<vector<neuron>>(n);
    weights = vector<vector<vector<double>>>(n - 1);
    size = vector<int>(n);
    for (int i = 0; i < n; i++) {
        size[i] = p[i];
        neurons[i] = vector<neuron>(p[i]);
        if (i < n - 1) {
            weights[i] = vector<vector<double>>(p[i]);
            for (int j = 0; j < p[i]; j++) {
                weights[i][j] = vector<double>(p[i + 1]);
                for (int k = 0; k < p[i + 1]; k++) {
                    fin >> weights[i][j][k];
                }
            }
        }
    }
}

NeuronNetwork::NeuronNetwork(int n, vector<int> p) {
    std::mt19937 mt(time(nullptr));
    layers = n;
    neurons = vector<vector<neuron>>(n);
    weights = vector<vector<vector<double>>>(n - 1);
    size = vector<int>(n);
    for (int i = 0; i < n; i++) {
        size[i] = p[i];
        neurons[i] = vector<neuron>(p[i]);
        if (i < n - 1) {
            weights[i] = vector<vector<double>>(p[i]);
            for (int j = 0; j < p[i]; j++) {
                weights[i][j] = vector<double>(p[i + 1]);
                for (int k = 0; k < p[i + 1]; k++) {
                    weights[i][j][k] = (((mt()) % 100)) * 0.01 / size[i];
                }
            }
        }
    }
}

void NeuronNetwork::setRandomInput() {
    std::mt19937 mt(time(nullptr));
    for (int i = 0; i < size[0]; i++) {
        neurons[0][i].value = (mt() % 256) / 255;
    }
}

void NeuronNetwork::set_input(vector<double> input) {// [0...255]
    for (int i = 0; i < size[0]; i++) {
        neurons[0][i].value = input[i];
    }
}

void NeuronNetwork::LayersCleaner(int LayerNumber, int start, int stop) {
    for (int i = start; i < stop; i++) {
        neurons[LayerNumber][i].value = 0;
    }
}

// Forward Feed helper to calculate
void NeuronNetwork::ForwardFeeder(int LayerNumber, int start, int stop) {
    for (int j = start; j < stop; j++) {
        for (int k = 0; k < size[LayerNumber - 1]; k++) {
            neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
        }
        neurons[LayerNumber][j].exponent();
    }
}

double NeuronNetwork::ForwardFeed() {
    for (int i = 1; i < layers; i++) {
        LayersCleaner(i, 0, size[i]);
    }
    double max = 0;
    double prediction = 0;
    for (int i = 0; i < size[layers - 1]; i++) {
        if (neurons[layers - 1][i].value > max) {
            max = neurons[layers - 1][i].value;
            prediction = static_cast<double>(i);
        }
    }
    return prediction;
}

void NeuronNetwork::ErrorCounter(int LayerNumber, int start, int stop, double prediction, double rresult, double lr) {
    if (LayerNumber == layers - 1) {
        for (int j = start; j < stop; j++) {
            if (j != int(rresult)) {
                neurons[LayerNumber][j].error = -pow((neurons[LayerNumber][j].value), 2);
            } else {
                neurons[LayerNumber][j].error = 1.0 - neurons[LayerNumber][j].value;
            }
        }
    } else {
        for (int j = start; j < stop; j++) {
            double error = 0.0;
            for (int k = 0; k < size[LayerNumber + 1]; k++) {
                error += neurons[LayerNumber + 1][k].error * weights[LayerNumber][j][k];
            }
            neurons[LayerNumber][j].error = error;
        }
    }
}

void NeuronNetwork::WeightsUpdater(int start, int stop, int LayerNum, int lr) {
    int i = LayerNum;
    for (int j = start; j < stop; j++) {
        for (int k = 0; k < size[i + 1]; k++) {
            weights[i][j][k] +=
                    lr * neurons[i + 1][k].error * SigmoidDerivative(neurons[i + 1][k].value) * neurons[i][j].value;
        }
    }
}

void NeuronNetwork::BackPropogation(double prediction, double rresult,
                                    double lr) {//Rresult must be any number < 0 if u dont know the right result
    for (int i = layers - 1; i > 0; i--) {
        if (i == layers - 1) {
            for (int j = 0; j < size[i]; j++) {
                if (j != int(rresult)) {
                    neurons[i][j].error = -pow((neurons[i][j].value), 2);
                } else {
                    neurons[i][j].error = 1.0 - neurons[i][j].value;
                }
            }
        } else {
            for (int j = 0; j < size[i]; j++) {
                double error = 0.0;
                for (int k = 0; k < size[i + 1]; k++) {
                    error += neurons[i + 1][k].error * weights[i][j][k];
                }
                neurons[i][j].error = error;
            }
        }
    }
    for (int i = 0; i < layers - 1; i++) {
        for (int j = 0; j < size[i]; j++) {
            for (int k = 0; k < size[i + 1]; k++) {
                weights[i][j][k] +=
                        lr * neurons[i + 1][k].error * SigmoidDerivative(neurons[i + 1][k].value) *
                        neurons[i][j].value;
            }
        }
    }
}

bool NeuronNetwork::SaveWeights() {
    std::ofstream fout;
    fout.open("weights.txt");
    for (int i = 0; i < layers; i++) {
        if (i < layers - 1) {
            for (int j = 0; j < size[i]; j++) {
                for (int k = 0; k < size[i + 1]; k++) {
                    fout << weights[i][j][k] << " ";
                }
            }
        }
    }
    fout.close();
    return true;
}

NeuronNetwork::NeuronNetwork() = default;
