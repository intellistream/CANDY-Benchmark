// Copyright (C) 2021 by the IntelliStream team
// (https://github.com/intellistream)

//
// Created by Shuhao Zhang on 20/8/21.
//

#ifndef SESAME_SRC_ENGINE_THREAD_HPP_
#define SESAME_SRC_ENGINE_THREAD_HPP_
#include <barrier>
#include <functional>
#include <memory>

namespace SESAME {
class SingleThread;
typedef std::shared_ptr<SingleThread> SingleThreadPtr;

class SingleThread {
private:
  std::shared_ptr<std::thread> ThreadPtr;
  int id;

public:
  void construct(std::function<void(void)> fun, int id);
  void join();
  int getID();
  int setID(int id);
};
} // namespace SESAME

#endif // SESAME_SRC_ENGINE_THREAD_HPP_
