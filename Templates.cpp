//
// Created by baymu on 01.11.2023.
//
#include "Templates.h"



Template::Template(vector<double> &h, vector<double> &v, vector<double> &e, vector<double> &l, vector<double> &r,
                   vector<double> &test_inp) {
    Template::h = h;
    Template::v = v;
    Template::e = e;
    Template::l = l;
    Template::r = r;
    Template::test_inp = test_inp;
}
