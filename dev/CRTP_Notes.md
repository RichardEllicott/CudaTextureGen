
# CRTP Notes

## CRTP Example

```CPP
template <typename Derived>
class CRTP_Base {
public:
void call_required() {
    // Forward to derived — must exist!
    static_cast<Derived *>(this)->required_fn();
}
};

// Derived class must implement required_fn
class CRTP_Test : public CRTP_Base<CRTP_Test> {
public:
void required_fn() {
    std::cout << "MyArray::required_fn implemented\n";
}
};
```
