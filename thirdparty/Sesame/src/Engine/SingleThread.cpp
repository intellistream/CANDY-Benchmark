// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/8/21.
//

#include <Engine/SingleThread.hpp>
void SESAME::SingleThread::construct(std::function<void(void)> fun, int id) {
  this->ThreadPtr = std::make_shared<std::thread>(fun);
  this->setID(id);
}
int SESAME::SingleThread::setID(int id) { return this->id = id; }
int SESAME::SingleThread::getID() { return this->id; }
void SESAME::SingleThread::join() { this->ThreadPtr->join(); }
