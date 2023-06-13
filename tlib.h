#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

//
// properties
//

#define TLIB_MEM_ALIGN 4      // bytes
#define TLIB_MAX_TENSOR_DIM 4 // 1D-4D support
#define TLIB_MAX_NAME 32      // 32 bytes string

//
// types and backends
//

enum tlib_types
{
    TLIB_TYPE_F32,
    TLIB_TYPE_COUNT
};

enum tlib_backend
{
    TLIB_BACKEND_CPU,
};
static const size_t TLIB_TYPE_SIZE[TLIB_TYPE_COUNT] = {
    [TLIB_TYPE_F32] = sizeof(float),
};

//
// context and memory related structs
//

struct tlib_init_params
{
    // memory pool
    size_t mem_size; // in bytes
    /*
    pool is allocated all at once.
    objects in runtime use this pool, with no allocations in runtime.
    */

    void *mem_buffer; // if NULL memory will be allocated internally
};

struct tlib_context
{
    size_t mem_size;
    void *mem_buffer;

    int n_objects;                    // number of objects in memory pool
    struct tlib_object *object_begin; // linked list of objects in memory pool
    struct tlib_object *object_end;
};

struct tlib_object
{
    size_t offset; // absolute offset in buffer. (assuming start of buffer to be 0)
    size_t size;   // size of the tensor object + data

    /*
    memory buffer layout
    |object|   |tensor object| |tensor data|   |object|
             ^                               ^
             |  <------ size (bytes) ------> |
           offset (bytes)                   next
    */

    struct tlib_object *next; // creates a linked list helping in retrieval of tensor from pool
};
static const size_t TLIB_OBJECT_SIZE = sizeof(struct tlib_object);

//
// tensor creation and manipulation
//

enum tlib_ops
{
    TLIB_OP_NONE,
    TLIB_OP_ADD,
    TLIB_OP_MUL,
};

struct tlib_tensor
{
    int n_dims;
    int64_t n_elements[TLIB_MAX_TENSOR_DIM]; // n elements per dimension - aka size.
    size_t n_bytes[TLIB_MAX_TENSOR_DIM];     // stride in bytes per dimension

    struct tlib_tensor *grad;
    struct tlib_tensor *src0;
    struct tlib_tensor *src1;

    bool is_param;

    void *data;
    enum tlib_types type;
    enum tlib_backend backend;
    enum tlib_ops op;

    char name[TLIB_MAX_NAME];
};
static const size_t TLIB_TENSOR_SIZE = sizeof(struct tlib_tensor);

struct tlib_tensor *tlib_new_tensor(struct tlib_context *ctx,
                                    enum tlib_types type,
                                    int n_dims,
                                    int64_t *ne);

struct tlib_tensor *tlib_new_tensor_impl(struct tlib_context *ctx,
                                         enum tlib_types type,
                                         int n_dims,
                                         int64_t *ne,
                                         void *data);
struct tlib_tensor *tlib_new_tensor_1d(struct tlib_context *ctx,
                                       enum tlib_types type,
                                       int64_t ne);

struct tlib_tensor *tlib_dup_tensor(struct tlib_context *ctx, struct tlib_tensor *x);

//
// supported tensor operations - unary and binary
//

struct tlib_tensor *tlib_add(struct tlib_context *ctx, struct tlib_tensor *a, struct tlib_tensor *b);
struct tlib_tensor *tlib_mul(struct tlib_context *ctx, struct tlib_tensor *a, struct tlib_tensor *b);

//
// init and memory management functions
//

struct tlib_context tlib_init(const struct tlib_init_params params);

//
// autograd
//

void tlib_set_param(struct tlib_context *ctx, struct tlib_tensor *const x);