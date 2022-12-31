#ifndef UNIFIED_MEMORY_HPP
#define UNIFIED_MEMORY_HPP

class UnifiedMemory {
public:
  void *operator new(size_t len);
  void operator delete(void *ptr);
};

#endif