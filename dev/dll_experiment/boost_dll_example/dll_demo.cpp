#include <boost/config.hpp> // for BOOST_SYMBOL_EXPORT
extern "C" BOOST_SYMBOL_EXPORT int add(int a, int b) {
    return a + b;
}
