//
// Created by baymu on 01.11.2023.
//

#include "DataReader.h"
void DataReader::readData( vector<data_block>& data) const {
    std::ifstream fin;
    fin.open("lib.txt");

    for (int i = 0; i < data.size(); i++) {
        char value;
        fin >> value;
        switch (value) {
            case 'h':
                data[i].type = 0;
                break;
            case 'v':
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
}
