/*

smart pointer wrapper based on Godot's Ref type, based on c++ shared_ptr

created as this allows a very easy instantiate()

created with CoPilot, uses mostly all shared_ptr features for copy/move/swap

*/
#pragma once
#include <memory>

namespace core {

template <typename T>
class Ref {

  public:
    std::shared_ptr<T> shared_ptr; //  seems to need to be public??

    // Godot-style: explicitly create the object
    template <typename... Args>
    void instantiate(Args &&...args) {
        shared_ptr = std::make_shared<T>(std::forward<Args>(args)...);
    }

    // Godot-style: drop the reference
    void unref() {
        shared_ptr.reset();
    }

    // Godot-style: check validity
    bool is_valid() const {
        return static_cast<bool>(shared_ptr);
    }

    bool is_null() const {
        return !shared_ptr;
    }

    // Access underlying object (no auto-create)
    T &get() {
        return *shared_ptr;
    }

    const T &get() const {
        return *shared_ptr;
    }

    // Pointer-like access
    T *operator->() { return shared_ptr.get(); }
    const T *operator->() const { return shared_ptr.get(); }

    operator bool() const { return is_valid(); }

    // Duplicate the underlying object (deep copy)
    Ref<T> duplicate() const {
        Ref<T> out;
        if (shared_ptr) {
            out.shared_ptr = std::make_shared<T>(*shared_ptr); // copy construct
        }
        return out;
    }

    // Create a new object with the same "shape" but empty
    // (requires T to have clone_empty() or similar)
    Ref<T> clone_empty() const {
        Ref<T> out;
        if (shared_ptr) {
            out.shared_ptr = shared_ptr->clone_empty();
        }
        return out;
    }
};

} // namespace core
