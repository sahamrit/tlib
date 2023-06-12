#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

//
// properties
//

#define TLIB_MEM_ALIGN 4 // bytes

//
// context and memory related structs
//

struct tlib_init_params
{
    // memory pool
    size_t mem_size; // in bytes
    // pool is allocated all at once.
    // objects in runtime use this pool, with no allocations in runtime.

    void *mem_buffer; // if NULL memory will be allocated internally
};

struct tlib_context
{
    size_t mem_size;
    void *mem_buffer;
};

struct tlib_object
{
    size_t offset;
    size_t size;
};

//
// init and memory management functions
//

struct tlib_context tlib_init(const struct tlib_init_params params);
