//
// Created by rubato on 31/5/24.
//

#ifndef ALG_H
#define ALG_H
#pragma once


#include <iostream>
#include <fstream>
//#include <RANIA/LSHAPGIndex/Preprocess.h>
//#include <RANIA/LSHAPGIndex/divGraph.h>
#include <CANDY/LSHAPGIndex/fastGraph.h>
#include <CANDY/LSHAPGIndex/Query.h>
#include <time.h>
//#include <RANIA/LSHAPGIndex/basis.h>
#include <sstream>

#if defined(unix) || defined(__unix__)
struct llt
{
	int date, h, m, s;
	llt(size_t diff) { set(diff); }
	void set(size_t diff)
	{
		date = diff / 86400;
		diff = diff % 86400;
		h = diff / 3600;
		diff = diff % 3600;
		m = diff / 60;
		s = diff % 60;
	}
};
#endif
template <class Graph>
void graphSearch(float c, int k, Graph* myGraph, Preprocess& prep, float beta, std::string& datasetName, std::string& data_fold, int qType);

void zlshKnn(float c, int k, e2lsh& myLsh, Preprocess& prep, float beta, std::string& datasetName, std::string& data_fold);
bool find_file(std::string&& file);

std::vector<queryN*> search_candy(float c, int k, divGraph* myGraph, Preprocess& prep, float beta, int qType);

#endif //ALG_H
