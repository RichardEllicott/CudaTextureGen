/*

custom global stuff

*/
#pragma once
#include <chrono>

namespace core {

class Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point t1{}, t2{};

  public:
    void mark_time() {
        t1 = t2;
        t2 = clock::now();
    }

    double elapsed_seconds() const {
        return std::chrono::duration<double>(t2 - t1).count();
    }

    Timer() { mark_time(); }
};

} // namespace core::util
