//
// Created by rubato on 31/5/24.
//

#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <CANDY.h>
#include <CANDY/LSHAPGIndex/Preprocess.h>
#include <iostream>
using namespace std;
using namespace INTELLI;
using namespace torch;
using namespace CANDY;
TEST_CASE("Test preprocess index", "[short]")
{
    Preprocess prep(10);
    auto data =new float[500];
    prep.insert_data(data, 25);
    prep.insert_data(data,25);
    for(int i=0; i<50; i++){
        for(int j=0; j<10; j++){
            printf(".2f ", prep.data.val[i][j];

        }
        printf("\n");

    }

}