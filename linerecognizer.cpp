#include<iostream>
#include<fstream>
#include<vector>
#include<random>

using std::vector;
using std::string;
using std::cout;
using std::cin;
using std::endl;
using std::size;

struct Neuron {
    double value;
    double error;

    void act() {
        value = (1 / (1 + pow(2.71828, -value)));
    }
};

int max(int a, int b) {
    if (a > b) return a;
    else return b;

}

class Network {
public:
    int layers;
    Neuron **neurons;
    double ***weights;
    int *size;

    double SigmoidPrime(double x) {
        if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
        double res = x * (1.0 - x);
        return res;
    }

    double Predict(double x) {
        if (x >= 0.8) {
            return 1;
        } else {
            return 0;
        }
    }

    void SetLayersNotStudy(int n, int *p, std::string filename) {
        std::ifstream fin;
        fin.open(filename);
        srand(time(0));
        layers = n;
        neurons = new Neuron *[n];
        weights = new double **[n - 1];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            size[i] = p[i];
            neurons[i] = new Neuron[p[i]];
            if (i < n - 1) {
                weights[i] = new double *[p[i]];
                for (int j = 0; j < p[i]; j++) {
                    weights[i][j] = new double[p[i + 1]];
                    for (int k = 0; k < p[i + 1]; k++) {
                        fin >> weights[i][j][k];
                    }
                }
            }
        }
    }

    void SetLayers(int n, vector<int> p) {
        srand(time(0));
        layers = n;
        neurons = new Neuron *[n];
        weights = new double **[n - 1];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            size[i] = p[i];
            neurons[i] = new Neuron[p[i]];
            if (i < n - 1) {
                weights[i] = new double *[p[i]];
                for (int j = 0; j < p[i]; j++) {
                    weights[i][j] = new double[p[i + 1]];
                    for (int k = 0; k < p[i + 1]; k++) {
                        weights[i][j][k] = (rand() % 100) * 0.01 / size[i];
                    }
                }
            }
        }
    }

    void SetRandomInput() {
        for (int i = 0; i < size[0]; i++) {
            neurons[0][i].value = (rand() % 256) / 255;
        }
    }

    void SetInput(vector<double> p) {
        for (int i = 0; i < size[0]; i++) {
            neurons[0][i].value = p[i];
        }
    }

    void Show() {
        setlocale(LC_ALL, "ru");
        for (int i = 0; i < layers; i++) {
            cout << size[i];
            if (i < layers - 1) {
                cout << " - ";
            }
        }
        cout << endl;
        for (int i = 0; i < layers; i++) {
            cout << "\n#Слой " << i + 1 << "\n\n";
            for (int j = 0; j < size[i]; j++) {
                cout << "Нейрон #" << j + 1 << ": \n";
                cout << "Значение: " << neurons[i][j].value << endl;
                if (i < layers - 1) {
                    cout << "Веса: \n";
                    for (int k = 0; k < size[i + 1]; k++) {
                        cout << "#" << k + 1 << ": ";
                        cout << weights[i][j][k] << endl;
                    }
                }
            }
        }
    }

    void LayersCleaner(int LayerNumber, int start, int stop) {
        for (int i = start; i < stop; i++) {
            neurons[LayerNumber][i].value = 0;
            // cout << "neurons[" << LayerNumber << "][" << i << "].value = " << neurons[LayerNumber][i].value << endl;
        }
    }

    void ForwardFeeder(int LayerNumber, int start, int stop) {
        for (int j = start; j < stop; j++) {
            for (int k = 0; k < size[LayerNumber - 1]; k++) {
                neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
            }
            // cout << "До активации: " << neurons[i][j].value << endl;
            neurons[LayerNumber][j].act();
        }
    }

    double ForwardFeed() {
        setlocale(LC_ALL, "ru");

        for (int i = 1; i < layers; i++) {
            LayersCleaner(i, 0, size[i]);
            ForwardFeeder(i, 0, size[i]);
        }
        double max = 0;
        double prediction = 0;
        for (int i = 0; i < size[layers - 1]; i++) {
            if (neurons[layers - 1][i].value > max) {
                max = neurons[layers - 1][i].value;
                prediction = i;
            }
        }
        return prediction;
    }

    void ErrorCounter(int LayerNumber, int start, int stop, double prediction, double rresult, double lr) {
        if (LayerNumber == layers - 1) {
            for (int j = start; j < stop; j++) {
                if (j != static_cast<int>(rresult)) {
                    neurons[LayerNumber][j].error = -pow(neurons[LayerNumber][j].value, 2);
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

    void WeightsUpdater(int start, int stop, int LayerNum, int lr) {
        int i = LayerNum;
        for (int j = start; j < stop; j++) {
            for (int k = 0; k < size[i + 1]; k++) {
                weights[i][j][k] +=
                        lr * neurons[i + 1][k].error * SigmoidPrime(neurons[i + 1][k].value) * neurons[i][j].value;
            }
        }
    }

    void BackPropagation(double prediction, double rresult, double lr) {
        // Rresult must be any number < 0 if you don't know the right result
        for (int i = layers - 1; i > 0; i--) {
            if (i == layers - 1) {
                for (int j = 0; j < size[i]; j++) {
                    if (j != static_cast<int>(rresult)) {
                        neurons[i][j].error = -pow(neurons[i][j].value, 2);
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

            for (int i = 0; i < layers - 1; i++) {
                for (int j = 0; j < size[i]; j++) {
                    for (int k = 0; k < size[i + 1]; k++) {
                        weights[i][j][k] += lr * neurons[i + 1][k].error * SigmoidPrime(neurons[i + 1][k].value) *
                                            neurons[i][j].value;
                    }
                }
            }
        }
    }

    bool SaveWeights() {
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
};

struct DataBlock {
    double data[64];
    int type; // 0 - horizontal, 1 - vertical
};

int main() {
    const int block_a_size = 8;
    const int n = 53;
    std::vector<DataBlock> data(n);

    std::ifstream fin;
    fin.open("dataset/lib.txt");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < block_a_size * block_a_size; j++) {
            fin >> data[i].data[j];
            std::cout << data[i].data[j] << " ";
        }
        std::cout << std::endl;
        char value;
        fin >> value;
        switch (value) {
            case 'h': // horizontal
                data[i].type = 0;
                break;
            case 'v': // vertical
                data[i].type = 1;
                break;
            case 'e':
                data[i].type = 2;
                break;
            case 'l':
                data[i].type = 3;
                break;
            case 'r':
                data[i].type = 4;
                break;
            case 'j':
                data[i].type = 9;
                break;
        }
    }

    std::vector<int> layers;
    layers = {int(pow(block_a_size, 2)), 64, 5};

    Network net;
    net.SetLayers(3, layers);
    bool study;
    std::cout << "To study (0/1)?";
    std::cin >> study;
    if (study) {
        double percentRight = 0;
        int epoch = 1;
        while (percentRight < 100) {
            std::cout << "Epoch: " << epoch << "\t";
            double rightAnswers = 0;
            for (int i = 0; i < n; i++) {
                std::vector<double> input(int(pow(block_a_size,
                2)));
                for (int j = 0; j < int(pow(block_a_size, 2)); j++) {
                    input[j] = data[i].data[j];
                }
                net.SetInput(input);
                double result = net.ForwardFeed();
                if (result != data[i].type) {
                    net.BackPropagation(result, data[i].type, 0.25);
                } else {
                    rightAnswers++;
                }
            }
            percentRight = rightAnswers / n * 100;
            std::cout << "Right answers: " << percentRight << " %\n";
            epoch++;
        }
        std::cout << percentRight << std::endl;
    }

    Image img;
    img.loadFromFile("dataset/image.jpg");
    string name = std::to_string(rand() % 10000);
    img.saveToFile("originals/" + name + ".jpg");
    Image res;
    const int height = img.getSize().y;
    const int width = img.getSize().x;
    res.create(width, height);
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
            res.setPixel(i, j, Color(255, 255, 255, 255));
    Texture texture;
    texture.loadFromImage(res);
    Sprite sprite;
    sprite.setTexture(texture);
    RenderWindow window(VideoMode(width, height), "NImages");

    vector<double> h(64);
    h = {
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,

    };
    vector<double> v(64);
    v = {
            0.000, 0.000, 0.275, 0.725, 0.690, 0.235, 0.000, 0.000,
            0.000, 0.000, 0.278, 0.737, 0.710, 0.239, 0.000, 0.000,
            0.000, 0.000, 0.286, 0.741, 0.702, 0.235, 0.000, 0.000,
            0.000, 0.000, 0.294, 0.745, 0.698, 0.227, 0.000, 0.000,
            0.000, 0.000, 0.278, 0.741, 0.706, 0.235, 0.000, 0.000,
            0.000, 0.000, 0.267, 0.733, 0.718, 0.251, 0.000, 0.000,
            0.000, 0.000, 0.267, 0.729, 0.718, 0.255, 0.000, 0.000,
            0.000, 0.000, 0.251, 0.671, 0.651, 0.220, 0.000, 0.000
    };

    vector<double> e(64);
    e = {
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
    };

    vector<double> l(64);
    l = {
            0.639, 0.514, 0.035, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.569, 0.671, 0.514, 0.035, 0.000, 0.000, 0.000, 0.000,
            0.067, 0.569, 0.671, 0.514, 0.035, 0.000, 0.000, 0.000,
            0.000, 0.067, 0.569, 0.671, 0.514, 0.035, 0.000, 0.000,
            0.000, 0.000, 0.067, 0.569, 0.671, 0.514, 0.035, 0.000,
            0.000, 0.000, 0.000, 0.067, 0.569, 0.671, 0.514, 0.035,
            0.000, 0.000, 0.000, 0.000, 0.067, 0.569, 0.671, 0.514,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.067, 0.569, 0.639

    };

    vector<double> r(64);
    r = {
            0.000, 0.000, 0.000, 0.000, 0.000, 0.067, 0.569, 0.639,
            0.000, 0.000, 0.000, 0.000, 0.067, 0.569, 0.671, 0.514,
            0.000, 0.000, 0.000, 0.067, 0.569, 0.671, 0.514, 0.035,
            0.000, 0.000, 0.067, 0.569, 0.671, 0.514, 0.035, 0.000,
            0.000, 0.067, 0.569, 0.671, 0.514, 0.035, 0.000, 0.000,
            0.067, 0.569, 0.671, 0.514, 0.035, 0.000, 0.000, 0.000,
            0.569, 0.671, 0.514, 0.035, 0.000, 0.000, 0.000, 0.000,
            0.639, 0.514, 0.035, 0.000, 0.000, 0.000, 0.000, 0.000

    };


    vector<double> test_inp(64);
    test_inp = {
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000

    };
    net.SetInput(test_inp);
    cout << "Prediction: " << net.ForwardFeed() << endl;

    for (int i = 0; i < height - block_a_size; i++) {
        for (int j = 0; j < width - block_a_size; j++) {
            vector<double> input(block_a_size * block_a_size);
            for (int y = 0; y < block_a_size; y++) {
                for (int x = 0; x < block_a_size; x++) {
                    input[y * block_a_size + x] = 1.0 - double(img.getPixel(j + x, i + y).r) / 255;
                }
            }

            net.SetInput(input);


            int result = net.ForwardFeed();


            if (result == 0) {
                for (int y = 0; y < block_a_size; y++) {
                    for (int x = 0; x < block_a_size; x++) {
                        int R = res.getPixel(j + x, i + y).r;
                        int A = res.getPixel(j + x, i + y).a;
                        res.setPixel(j + x, i + y,
                                     Color(255 - h[block_a_size * y + x] * 255, 255 - h[block_a_size * y + x] * 255,
                                           255 - h[block_a_size * y + x] * 255, 255));
                    }
                }

            }
            if (result == 1) {
                for (int y = 0; y < block_a_size; y++) {
                    for (int x = 0; x < block_a_size; x++) {
                        int R = res.getPixel(j + x, i + y).r;
                        int A = res.getPixel(j + x, i + y).a;

                        res.setPixel(j + x, i + y,
                                     Color(255 - v[block_a_size * y + x] * 255, 255 - v[block_a_size * y + x] * 255,
                                           255 - v[block_a_size * y + x] * 255, 255));
                    }
                }
            }

            if (result == 2) {
                for (int y = 0; y < block_a_size; y++) {
                    for (int x = 0; x < block_a_size; x++) {
                        int R = res.getPixel(j + x, i + y).r;
                        int A = res.getPixel(j + x, i + y).a;

                        res.setPixel(j + x, i + y,
                                     Color(255 - e[block_a_size * y + x] * 255, 255 - e[block_a_size * y + x] * 255,
                                           255 - e[block_a_size * y + x] * 255, 255));
                    }
                }

            }

            if (result == 3) {
                for (int y = 0; y < block_a_size; y++) {
                    for (int x = 0; x < block_a_size; x++) {
                        int R = res.getPixel(j + x, i + y).r;
                        int A = res.getPixel(j + x, i + y).a;

                        res.setPixel(j + x, i + y,
                                     Color(255 - l[block_a_size * y + x] * 255, 255 - l[block_a_size * y + x] * 255,
                                           255 - l[block_a_size * y + x] * 255, 255));
                    }
                }
            }
            if (result == 4) {
                for (int y = 0; y < block_a_size; y++) {
                    for (int x = 0; x < block_a_size; x++) {
                        int R = res.getPixel(j + x, i + y).r;
                        int A = res.getPixel(j + x, i + y).a;

                        res.setPixel(j + x, i + y,
                                     Color(255 - r[block_a_size * y + x] * 255, 255 - r[block_a_size * y + x] * 255,
                                           255 - r[block_a_size * y + x] * 255, 255));
                    }
                }
            }


        }
        texture.loadFromImage(res);
        sprite.setTexture(texture);
        window.clear();
        window.draw(sprite);
        window.display();
        cout << double(i) / double(height - block_a_size) * 100 << " % " << endl;
    }
    res.saveToFile("result2.jpg");
    res.saveToFile("library/" + name + ".jpg");
    return 0;
}
