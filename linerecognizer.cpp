#include "DataReader.h"
#include "NeuronNetwork.h"
#include "Templates.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <random>
#include <vector>
#pragma once

void toStudy(int n, int block_a_size, vector<data_block> &data, NeuronNetwork &net) {
    double percentRight = 0;
    int epoch = 1;
    while (percentRight < 100) {
        cout << "Epoch: " << epoch << "\t";
        double rightAnswers = 0;
        for (int i = 0; i < n; i++) {
            vector<double> input(int(pow(block_a_size,
                                         2)));
            for (int j = 0; j < int(pow(block_a_size, 2)); j++) {
                input[j] = data[i].data[j];
            }
            net.set_input(input);
            double result = net.ForwardFeed();
            if (result != data[i].type) {
                net.BackPropogation(result, data[i].type, 0.25);
            } else {
                rightAnswers++;
            }
        }
        percentRight = rightAnswers / n * 100;
        cout << "Right answers: " << percentRight << " %\n";
        epoch++;
    }
    cout << percentRight << endl;
}
int main() {
    std::mt19937 mt(time(nullptr));
    const int block_a_size = 8;
    int n;
    std::cin >> n;
    DataReader dataReader;
    vector<data_block> data(n);
    dataReader.readData(data);
    vector<int> layers;
    layers = {int(pow(block_a_size, 2)), 64, 5};

    NeuronNetwork net(3, layers);
    bool study;
    cout << "To toStudy(0/1)?";
    cin >> study;
    if (study) {
        toStudy(n, block_a_size, data, net);
    }
    sf::Image img;
    img.loadFromFile("C:\\cppProjects\\Tinkoff\\MLbeginner\\dataset\\image.jpg");
    string name = std::to_string(mt() % 10000);
    img.saveToFile(name + ".jpg");
    sf::Image res;
    const size_t height = img.getSize().y;
    const size_t width = img.getSize().x;
    res.create(width, height);
    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
            res.setPixel(i, j, sf::Color(255, 255, 255, 255));
    sf::Texture t;
    t.loadFromImage(res);
    sf::Sprite s;
    s.setTexture(t);
    sf::RenderWindow window(sf::VideoMode(width, height), "NImages");

    // horizonal
    vector<double> h(64);
    h = {
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.498, 0.498, 0.000, 0.000, 0.000};
    //vertical
    vector<double> v(64);
    v = {
            0.000, 0.000, 0.275, 0.725, 0.690, 0.235, 0.000, 0.000,
            0.000, 0.000, 0.278, 0.737, 0.710, 0.239, 0.000, 0.000,
            0.000, 0.000, 0.286, 0.741, 0.702, 0.235, 0.000, 0.000,
            0.000, 0.000, 0.294, 0.745, 0.698, 0.227, 0.000, 0.000,
            0.000, 0.000, 0.278, 0.741, 0.706, 0.235, 0.000, 0.000,
            0.000, 0.000, 0.267, 0.733, 0.718, 0.251, 0.000, 0.000,
            0.000, 0.000, 0.267, 0.729, 0.718, 0.255, 0.000, 0.000,
            0.000, 0.000, 0.251, 0.671, 0.651, 0.220, 0.000, 0.000};
    //empty
    vector<double> e(64);
    e = {
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
    // left diag
    vector<double> l(64);
    l = {
            0.639, 0.514, 0.035, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.569, 0.671, 0.514, 0.035, 0.000, 0.000, 0.000, 0.000,
            0.067, 0.569, 0.671, 0.514, 0.035, 0.000, 0.000, 0.000,
            0.000, 0.067, 0.569, 0.671, 0.514, 0.035, 0.000, 0.000,
            0.000, 0.000, 0.067, 0.569, 0.671, 0.514, 0.035, 0.000,
            0.000, 0.000, 0.000, 0.067, 0.569, 0.671, 0.514, 0.035,
            0.000, 0.000, 0.000, 0.000, 0.067, 0.569, 0.671, 0.514,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.067, 0.569, 0.639};
    // right diag
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
    Template temp(h, v, e, l, r, test_inp);
    net.set_input(test_inp);
    cout << "Prediction: " << net.ForwardFeed() << endl;

    for (int i = 0; i < height - block_a_size; i++) {
        for (int j = 0; j < width - block_a_size; j++) {
            vector<double> input(block_a_size * block_a_size);
            for (int y = 0; y < block_a_size; y++) {
                for (int x = 0; x < block_a_size; x++) {
                    input[y * block_a_size + x] = 1.0 - double(img.getPixel(j + x, i + y).r) / 255;
                }
            }
            net.set_input(input);
            int result = net.ForwardFeed();
            if (result == 0) {
                for (int y = 0; y < block_a_size; y++) {
                    for (int x = 0; x < block_a_size; x++) {
                        int R = res.getPixel(j + x, i + y).r;
                        int A = res.getPixel(j + x, i + y).a;
                        res.setPixel(j + x, i + y,
                                     sf::Color(255 - h[block_a_size * y + x] * 255, 255 - h[block_a_size * y + x] * 255,
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
                                     sf::Color(255 - v[block_a_size * y + x] * 255, 255 - v[block_a_size * y + x] * 255,
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
                                     sf::Color(255 - e[block_a_size * y + x] * 255, 255 - e[block_a_size * y + x] * 255,
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
                                     sf::Color(255 - l[block_a_size * y + x] * 255, 255 - l[block_a_size * y + x] * 255,
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
                                     sf::Color(255 - r[block_a_size * y + x] * 255, 255 - r[block_a_size * y + x] * 255,
                                               255 - r[block_a_size * y + x] * 255, 255));
                    }
                }
            }
        }
        t.loadFromImage(res);
        s.setTexture(t);
        window.clear();
        window.draw(s);
        window.display();
        cout << double(i) / double(height - block_a_size) * 100 << " % " << endl;
    }
    res.saveToFile("result2.jpg");
    res.saveToFile("library/" + name + ".jpg");
    return 0;
}
