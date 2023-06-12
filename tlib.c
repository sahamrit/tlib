#include "tlib.h"

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <inttypes.h>

struct tlib_context tlib_init(const struct tlib_init_params params)
{
    size_t mem_size_aligned = (params.mem_size + TLIB_MEM_ALIGN - 1) & (~(TLIB_MEM_ALIGN - 1));
    void *mem_buffer = NULL;

    // Perform aligned memory allocation
    mem_buffer = _aligned_malloc(mem_size_aligned, TLIB_MEM_ALIGN);
    // TODO specific with windows

    if (mem_buffer == NULL)
    {
        fprintf(stderr, "STDERR: %s: Aligned memory allocation failed, size requested - %d bytes.\n", __FILE__, mem_size_aligned);
        assert(false);
        return;
    }

    struct tlib_context ctx = (struct tlib_context){
        /*mem_size =*/mem_size_aligned,
        /*mem_buffer =*/mem_buffer,
        /*n_objects =*/0,
        /*object_begin =*/NULL,
        /*object_end =*/NULL,
    };
    return ctx;
}

struct tlib_tensor *tlib_new_tensor_impl(struct tlib_context *ctx,
                                         enum tlib_types type,
                                         int n_dims,
                                         int64_t *ne,
                                         void *data)
{
    struct tlib_object *const cur_obj = ctx->object_end;
    const size_t curr_offset = cur_obj == NULL ? 0 : cur_obj->offset;
    const size_t curr_size = cur_obj == NULL ? 0 : cur_obj->size;
    size_t curr_end = curr_offset + curr_size;
    // current end of the memory pool

    // calculate memory requirement for the new tensor
    // new tensors are always contiguous
    size_t size_needed = TLIB_TYPE_SIZE[type];
    for (int i = 0; i < n_dims; i++)
    {
        size_needed *= ne[i];
    }

    struct tlib_object *const new_obj = (struct tlib_object *)(ctx->mem_buffer + curr_end);
    size_needed += TLIB_TENSOR_SIZE;

    if (curr_end + size_needed + TLIB_OBJECT_SIZE > ctx->mem_size)
    {
        fprintf(stderr, "STDERR: %s: pool memory space exceeded (needed %zu, avaialable %zu)",
                __FILE__, curr_end + size_needed + TLIB_OBJECT_SIZE, ctx->mem_size);
        assert(false);
        return;
    }

    *new_obj = (struct tlib_object){
        /*offset =*/curr_end + TLIB_OBJECT_SIZE,
        /*size =*/size_needed,
        /*next =*/NULL,
    };

    if (ctx->object_begin == NULL)
    {
        ctx->object_begin = new_obj;
    }
    else
    {
        cur_obj->next = new_obj;
    }

    struct tlib_tensor *const new_tensor = (struct tlib_tensor *)(ctx->mem_buffer + new_obj->offset);

    *new_tensor = (struct tlib_tensor){
        /*ndims =*/n_dims,
        /*n_elements =*/{1, 1, 1, 1},
        /*n_bytes =*/{0, 0, 0, 0},
        /*grad =*/NULL,
        /*src0 =*/NULL,
        /*src1 =*/NULL,
        /*is_param =*/false,
        /*data =*/NULL, // TODO
        /*type =*/type,
        /*backend =*/TLIB_BACKEND_CPU,
        /*op =*/TLIB_OP_NONE,
        /*name =*/{0},
    };

    for (int i = 0; i < n_dims; i++)
    {
        new_tensor->n_elements[i] = ne[i];
    }

    ctx->n_objects++;

    new_tensor->n_bytes[n_dims - 1] = TLIB_TYPE_SIZE[type];
    for (int i = n_dims - 2; i >= 0; i--)
    {
        new_tensor->n_bytes[i] = new_tensor->n_bytes[i + 1] * new_tensor->n_elements[i];
    }

    return new_tensor;
}
struct tlib_tensor *tlib_new_tensor_1d(struct tlib_context *ctx,
                                       enum tlib_types type,
                                       int64_t ne)
{
    return tlib_new_tensor(ctx, type, 1, &ne);
}

struct tlib_tensor *tlib_new_tensor(struct tlib_context *ctx,
                                    enum tlib_types type,
                                    int n_dims,
                                    int64_t *ne)
{
    return tlib_new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

int main()
{
    struct tlib_init_params params = (struct tlib_init_params){
        /*mem_size =*/16 * 1024 * 1024,
        /*mem_buffer =*/NULL,

    };

    struct tlib_context ctx;
    ctx = tlib_init(params);

    fprintf(stdout, "STDOUT: %s: Context initialised with mem size: %d bytes", __FILE__, ctx.mem_size);

    struct tlib_tensor *x = tlib_new_tensor_1d(&ctx, TLIB_TYPE_F32, 10);

    fprintf(stdout, "STDOUT: %s: Allocated tensor of shape (%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 ") and stride (%zu, %zu, %zu, %zu)",
            __FILE__, x->n_elements[0], x->n_elements[1], x->n_elements[2], x->n_elements[3],
            x->n_bytes[0], x->n_bytes[1], x->n_bytes[2], x->n_bytes[3]);
}