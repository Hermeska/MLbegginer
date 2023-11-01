//
// Created by baymu on 01.11.2023.
//

#ifndef MLBEGINNER_DATAREADER_H
#define MLBEGINNER_DATAREADER_H

#include <vector>
#include <fstream>

using std::vector;
struct data_block {
    double data[64];
    int type;//0 - horizontal, 1 - vertical, etc...
};
class DataReader {
public:
    void readData( vector<data_block>& data) const;
};


#endif //MLBEGINNER_DATAREADER_H
