#ifndef BASE62_H
#define BASE62_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

namespace base62 {

inline constexpr std::string_view kAlphabet =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"; // for mapping from 0-61 to alphanumeric

inline std::string Encode(std::uint64_t n) {
  if (n == 0) return std::string(1, kAlphabet[0]);
  std::string out;
  while (n > 0) {
    out.push_back(kAlphabet[n % 62]);
    n /= 62;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

}  // namespace base62

#endif // BASE62_H
