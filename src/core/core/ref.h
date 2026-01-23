/*

smart pointer wrapper based on Godot's Ref type, based on c++ shared_ptr

created as this allows a very easy instantiate(), with shared_ptr's alone i required a macro

copy/move/swap is implicit


custom type_caster in nanobind_helper allows python to work with this same as it does with shared_ptr


created with CoPilot

*/
#pragma once
#include <memory>

namespace core {

#define REF_REFACTOR_TYPE 1 // 0 = orginal, 1 = new virtual
#if REF_REFACTOR_TYPE == 0

class RefBase {
  public:
    void instantiate_if_null() {}
    bool is_null() const { return true; }
    bool is_valid() const { return false; }
    operator bool() const { return false; }
};

template <typename T>
class Ref : RefBase {
  public:
    std::shared_ptr<T> shared_ptr; // i need to leave this public to allow the type caster to see it

    // Godot-style: explicitly create the object
    template <typename... Args>
    void instantiate(Args &&...args) {
        shared_ptr = std::make_shared<T>(std::forward<Args>(args)...);
    }

    // Godot-style: drop the reference
    void unref() { shared_ptr.reset(); }

    // Godot-style: check validity
    bool is_valid() const { return static_cast<bool>(shared_ptr); }
    bool is_null() const { return !is_valid(); }

    // Instantiate if not already (with optional args)
    template <typename... Args>
    void instantiate_if_null(Args &&...args) {
        if (is_null()) {
            instantiate(std::forward<Args>(args)...);
        }
    }

    // Access underlying object (no auto-create)
    T &get() { return *shared_ptr; }

    const T &get() const { return *shared_ptr; }

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

#elif REF_REFACTOR_TYPE == 1

class RefBase {
  public:
    virtual ~RefBase() = default;

    virtual void instantiate_if_null() = 0;
    virtual bool is_valid() const = 0;
    virtual bool is_null() const = 0;
    virtual void *raw_ptr() const = 0;
};

template <typename T>
class Ref : RefBase {
  public:
    // Main shared_ptr
    std::shared_ptr<T> shared_ptr;

    // Explicitly create the object
    template <typename... Args>
    void instantiate(Args &&...args) {
        shared_ptr = std::make_shared<T>(std::forward<Args>(args)...);
    }

    // Drop the reference
    void unref() { shared_ptr.reset(); }

    // If valid
    bool is_valid() const override { return static_cast<bool>(shared_ptr); }

    // If null
    bool is_null() const override { return !is_valid(); }

    // Instantiate if not already
    void instantiate_if_null() override {
        if (!shared_ptr) {
            shared_ptr = std::make_shared<T>();
        }
    }

    // Instantiate if not already (with optional args)
    template <typename... Args>
    void instantiate_if_null(Args &&...args) {
        if (!shared_ptr) {
            shared_ptr = std::make_shared<T>(std::forward<Args>(args)...);
        }
    }

    // Raw pointer or void*
    void *raw_ptr() const override { return shared_ptr.get(); }

    // Access underlying object (no auto-create)
    T &get() { return *shared_ptr; }

    const T &get() const { return *shared_ptr; }

    // Pointer-like access
    T *operator->() { return shared_ptr.get(); }
    const T *operator->() const { return shared_ptr.get(); }

    // allow checking as bool
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

#endif
#undef REF_REFACTOR_TYPE

} // namespace core
