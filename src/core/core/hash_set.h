/*

simple wrapper for unordered_set

*/

#pragma once

#include <stdexcept>
#include <unordered_set>

namespace core {

template <typename T>
class HashSet {
    std::unordered_set<T> data_;

  public:
    void add(const T &value) {
        data_.insert(value);
    }

    bool contains(const T &value) const {
        return data_.count(value) > 0;
    }

    void remove(const T &value) {
        auto it = data_.find(value);
        if (it == data_.end()) {
            throw std::runtime_error("HashSet: value not present");
        }
        data_.erase(it);
    }

    void discard(const T &value) {
        data_.erase(value); // no throw
    }

    void clear() {
        data_.clear();
    }

    std::size_t size() const {
        return data_.size();
    }

    bool empty() const {
        return data_.empty();
    }

    // iteration support
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
};

} // namespace core