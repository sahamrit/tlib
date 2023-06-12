#include "tlib.h"

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

struct tlib_context tlib_init(const struct tlib_init_params params)
{
    size_t mem_size_aligned = (params.mem_size + TLIB_MEM_ALIGN - 1) & (~(TLIB_MEM_ALIGN - 1));
    void *mem_buffer = NULL;

    // Perform aligned memory allocation
    mem_buffer = _aligned_malloc(mem_size_aligned, TLIB_MEM_ALIGN);
    // TODO specific with windows

    if (mem_buffer == NULL)
    {
        fprintf(stderr, "Aligned memory allocation failed, size requested - %d bytes.\n", mem_size_aligned);
        return;
    }

    struct tlib_context ctx = (struct tlib_context){
        /*mem_size*/ mem_size_aligned,
        /*mem_buffer*/ mem_buffer,
    };
    return ctx;
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
}