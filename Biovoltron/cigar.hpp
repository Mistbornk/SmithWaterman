#pragma once

#include <algorithm>
#include <istream>
#include <ostream>
#include <vector>
#include <string>
#include <string_view>

namespace biovoltron {

/**
 * @ingroup file_io
 * @brief A format to represent an alignment of a sequence to a reference genome by encoding a sequence of events.
 */
struct Cigar {
  /**
   * @brief Element of a cigar string describing a continuous and consistent operation for a segment of residues.
   *
   * Example
   * ```cpp
   * #include <iostream>
   * #include <biovoltron/file_io/cigar.hpp>
   * 
   * int main() {
   *   using namespace biovoltron;
   *   auto element1 = Cigar::Element{ 5, 'M'};
   *   std::cout << std::string(element1) << std::endl;
   *   // Output: 5M
   *
   *   auto element2 = Cigar::Element{ 5, 'M'};
   *   std::cout << std::move(element1 == element2) << std::endl;
   *   // Output: 1
   *
   *   element2.op = 'I';
   *   std::cout << std::move(element1 == element2) << std::endl;
   *   // Output: 0
   * }
   * ```
   */
  struct Element {
    /**
     * @brief Size of continuous residues contained in this element with a specific operation
     */
    unsigned size{};

    /**
     * @brief CIGAR operation. M for alignmet match, I for insertion, and so on.
     */
    char op{};

    Element(unsigned s, char o) : size(s), op(o) {}
    /**
     * @brief Implicitly convert Element into `string` type.
     * 
     * @return the string consists of size and op of `this`
     */

    operator std::string() const { return std::to_string(size) + op; }

    /**
     * @brief Overload == operator for `this` object to compare with another Element object.
     * 
     * @param Element object of this type.
     * @return true if the two Elements objects are the same, false otherwise.
     */
    bool operator==(const Element& other) const noexcept {
      return size == other.size && op == other.op;
    }
  };

 private:
  std::vector<Element> elements;
  static auto
  to_elements(std::string_view cigar_string) {
    if (cigar_string == "*")
      return std::vector<Element>{};
    auto elements = std::vector<Element>{};
    for (auto i = 0u; i < cigar_string.size(); i++) {
      auto size = cigar_string[i] - '0';
      for (i++; std::isdigit(cigar_string[i]); i++)
        size = size * 10 + cigar_string[i] - '0';
      elements.emplace_back(size, cigar_string[i]);
    }
    return elements;
  }

 public:
  /**
   * @brief Default constructor.
   */
  Cigar() = default;

  /**
   * @brief Construct `this` from type convertible into *string_view*.
   * 
   * @param cigar_string cigar string with type convertible into *string_view*
   */
  template <typename T>
  explicit Cigar(const T& cigar_string)
  : elements(to_elements(std::string_view(cigar_string))) { }

  /**
   * @brief Overload assignent operator to build up Cigar::Element from type being convertible into *string_view*.
   * 
   * @param cigar_string cigar string with type convertible into *string_view*
   * @return pointer to `this`
   */
  template <typename T>
  auto& operator=(const T& cigar_string) {
    elements = to_elements(std::string_view(cigar_string));
    return *this;
  }

  /**
   * @brief Merge every continuous elements with identical Element::op into a single element.
   */
  auto
  compact() {
    if (elements.size() <= 1)
      return;
    auto res = std::vector<Element>{};
    auto [last_size, last_op] = elements.front();
    for (size_t idx = 1; idx < elements.size(); ++idx) {
      auto cur_size = elements[idx].size;
      auto cur_op = elements[idx].op;
      if (cur_op == last_op)
        last_size += cur_size;
      else {
        res.emplace_back(last_size, last_op);
        last_size = cur_size;
        last_op = cur_op;
      }
    }
    res.emplace_back(last_size, last_op);
    elements.swap(res);
  }

  /**
   * @brief Append elements by an element.
   */
  auto
  emplace_back(unsigned size, char op) {
    elements.emplace_back(size, op);
  }

  /**
   * @brief Append elements by an element.
   */
  auto
  push_back(Element element) {
    elements.push_back(element);
  }

  /**
   * @brief Append elements by elements of an other Cigar.
   */
  auto
  append(const Cigar& other) {
    for (const auto element : other.elements) elements.push_back(element);
  }

  /**
   * @brief Swap elements with elements of another Cigar.
   */
  auto
  swap(Cigar& other) {
    elements.swap(other.elements);
  }

  /**
   * @brief Accumulate Element::size of the elements with Element::op being `M`, `D`, `N`, `=`, or `X`.
   * 
   * @return sum of Element::size of the elements with Element::op `M`, `D`, `N`, `=`, or `X`
   */
  auto
  ref_size() const noexcept {
    auto ref_size = 0;
    for (auto [size, op] : elements) {
      switch (op) {
        case 'M':
        case 'D':
        case 'N':
        case '=':
        case 'X':
          ref_size += size;
          break;
        default:
          break;
      }
    }
    return ref_size;
  }

  /**
   * @brief Accumulate Element::size of the elements with Element::op being `M`, `I`, `S`, `=`, or `X`.
   * 
   * @return sum of Element::size of the elements with Element::op being `M`, `I`, `S`, `=`, or `X`
   */
  auto
  read_size() const noexcept {
    auto read_size = 0;
    for (auto [size, op] : elements) {
      switch (op) {
        case 'M':
        case 'I':
        case 'S':
        case '=':
        case 'X':
          read_size += size;
          break;
        default:
          break;
      }
    }
    return read_size;
  }

  /**
   * @brief Accumulate Element::size of the elements with Element::op being `S` or `H`.
   * 
   * @return sum of Element::size of the elements with Element::op being `S` or `H`
   */
  auto
  clip_size() const noexcept {
    auto clip_size = 0;
    for (auto [size, op] : elements) {
      switch (op) {
        case 'S':
        case 'H':
          clip_size += size;
          break;
        default:
          break;
      }
    }
    return clip_size;
  }

  /**
   * @brief Get the first element of elements.
   * 
   * @return the *iterator* pointing to the first element of elements
   */
  auto
  begin() {
    return elements.begin();
  }

  /**
   * @brief Get the first element of elements.
   * 
   * @return the *const iterator* pointing to the first element of elements
   */
  auto
  begin() const {
    return elements.begin();
  }

  /**
   * @brief Get the last element of elements.
   * 
   * @return the *iterator* pointing to the last element of elements
   */
  auto
  end() {
    return elements.end();
  }

  /**
   * @brief Get the last element of elements.
   * 
   * @return the *const iterator* pointing to the last element of elements
   */
  auto
  end() const {
    return elements.end();
  }

  /**
   * @brief Get the first element of elements.
   * 
   * @return the *reference* to the first element of elements
   */
  auto&
  front() {
    return elements.front();
  }

  /**
   * @brief Get the first element of elements.
   * 
   * @return the *const reference* to the first element of elements
   */
  auto&
  front() const {
    return elements.front();
  }

  /**
   * @brief Get the last element of elements.
   * 
   * @return the *reference* to the last element of elements
   */
  auto&
  back() {
    return elements.back();
  }

  /**
   * @brief Get the last element of elements.
   * 
   * @return the *const reference* to the last element of elements
   */
  auto&
  back() const {
    return elements.back();
  }

  /**
   * @brief Overload subscript operator to access the ith element of elements.
   *
   * @param i index of the accessed
   * @return the *reference* to the ith element of elements
   */
  auto&
  operator[](unsigned i) {
    return elements[i];
  }

  /**
   * @brief Overload subscript operator to access the ith element of elements.
   *
   * @param i the index of the accessed
   * @return the *const reference* to the ith element of elements
   */
  auto&
  operator[](unsigned i) const {
    return elements[i];
  }

  /**
   * @brief Implicitly convert `this` to string type.
   *
   * @return the string converted from `this`
   */
  operator std::string() const {
    auto cigar_string = std::string{};
    for (const auto element : elements) cigar_string += element;
    return cigar_string;
  }

  /**
   * @brief Remove the first element of elements.
   */
  auto
  pop_front() {
    elements.erase(elements.begin());
  }

  /**
   * @brief Remove the last element of elements.
   */
  auto
  pop_back() {
    elements.pop_back();
  }

  /**
   * @brief Reverse elements in elements.
   */
  auto reverse() { std::reverse(elements.begin(), elements.end()); }

  /**
   * @brief Check if operation of interest is contained in `this`.
   *
   * @param key operation of interest
   * @return true if key is contained in elements, false otherwise
   */
  auto
  contains(char key) const noexcept {
    for (auto [size, op] : elements)
      if (key == op)
        return true;
    return false;
  }

  /**
   * @brief Check if any operation of interest ones is contained in `this`.
   *
   * @param keys operations of interest
   * @return true if any key in keys is contained in elements, false otherwise
   */
  auto
  contains(std::string_view keys) const noexcept {
    for (auto [size, op] : elements)
      for (auto key : keys)
        if (key == op)
          return true;
    return false;
  }

  /**
   * @brief Get the size of elements of `this`.
   *
   * @return the size of elements of `this`
   */
  auto
  size() const noexcept {
    return elements.size();
  }

  auto clear() noexcept {
    elements.clear();
  }

  /**
   * @brief Overload == operator for `this` to compare with another Cigar object.
   *
   * @param Cigar the Cigar object to be compared
   * @return true if the two Cigar are the same, false otherwise
   */
  bool operator==(const Cigar& other) const noexcept {
    return elements == other.elements;
  }

  /**
   * @brief Overload << operator to write the size and the op of each element in the given Cigar object on the given ostream.
   *
   * @param os an ostream object
   * @param cigar the Cigar object to be written
   * @return the ostream on which the size and the op of each element in the Cigar object are written
   */
  friend auto&
  operator<<(std::ostream& os, const Cigar& cigar) {
    if (cigar.size() == 0)
      return os << "*";
    for (const auto [size, op] : cigar) os << size << op;
    return os;
  }

  /**
   * @brief Overload >> operator to read from the istream into a string used to construct the given Cigar object.
   *
   * @param os an istream object
   * @param cigar the Cigar object to be read into
   * @return the istream from which the istream is read.
   */
  friend auto&
  operator>>(std::istream& is, Cigar& cigar) {
    std::string cigar_string;
    is >> cigar_string;
    cigar = cigar_string;
    return is;
  }
};

}  // namespace biovoltron