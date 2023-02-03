#ifndef UNIFIED_MEMORY_H
#define UNIFIED_MEMORY_H

class UnifiedMemory {
public:
  void *operator new(size_t len);
  void operator delete(void *ptr);
};

#endif