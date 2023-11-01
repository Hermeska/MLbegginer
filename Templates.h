//
// Created by baymu on 01.11.2023.
//
#include <vector>
using std::vector;
#ifndef MLBEGINNER_TEMPLATES_H
#define MLBEGINNER_TEMPLATES_H
// works only on 64
class Template {
     vector<double> h;
     vector<double> v;
     vector<double> e;
     vector<double> l;
     vector<double> r;
     vector<double> test_inp;

public:
    Template(vector<double> &h, vector<double> &v, vector<double> &e, vector<double> &l, vector<double> &r, vector<double> &test_inp);
};

#endif//MLBEGINNER_TEMPLATES_H
