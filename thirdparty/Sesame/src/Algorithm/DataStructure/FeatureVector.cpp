//
// Created by tuidan on 2021/8/24.
//

#include <Algorithm/DataStructure/FeatureVector.hpp>

SESAME::CF::CF() { this->NumberOfNodes = 0; }
SESAME::CF::~CF() {}

int SESAME::CF::getN() const { return this->NumberOfNodes; }

void SESAME::CF::setN(int n) { this->NumberOfNodes = n; }
std::vector<double> SESAME::CF::getLS() const { return this->LS; }

std::vector<double> SESAME::CF::getSS() const { return this->SS; }

double SESAME::CF::getLSItem(int index) const {
  return this->getLS().at(index);
}

double SESAME::CF::getSSItem(int index) const {
  return this->getSS().at(index);
}

void SESAME::CF::setLS(std::vector<double> &newLs) {
  if (this->getLS().empty()) {
    for (double newL : newLs) {
      this->LS.push_back(newL);
    }
  } else {
    if (this->getLS().size() != newLs.size()) {
      std::cout << "Size Error: CF's LS size: " << this->getLS().size()
                << ", newLS's size: " << newLs.size();
    } else {
      for (int i = 0; i < newLs.size(); i++) {
        this->LS[i] = newLs[i];
      }
    }
  }
}

void SESAME::CF::setSS(std::vector<double> &newSs) {
  if (this->getSS().empty()) {
    for (double newS : newSs) {
      this->SS.push_back(newS);
    }
  } else {
    if (this->SS.size() != newSs.size()) {
      std::cout << "Size Error: CF's SS size: " << this->getSS().size()
                << ", newSs's size: " << newSs.size();
    } else {
      for (int i = 0; i < newSs.size(); i++) {
        this->SS[i] = newSs[i];
      }
    }
  }
}
SESAME::CFPtr SESAME::CF::copy() { return std::make_shared<CF>(*this); }
int SESAME::CF::getIndex() { return this->index; }
void SESAME::CF::setIndex(int id) { this->index = id; }
