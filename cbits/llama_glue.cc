#include "llama.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

llama_model* hs_llama_load_model(const char* filepath, int n_gpu_layers) {
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model* model = llama_load_model_from_file(filepath, model_params);
    return model;
}

void hs_llama_free_model(llama_model* model) {
    if (model) {
        llama_free_model(model);
    }
}

int hs_llama_read_context_length_from_metadata(llama_model* model) {
    assert(model);

    char arch_name[513];
    memset(arch_name, 0, sizeof(arch_name));
    int32_t result = llama_model_meta_val_str(model, "general.architecture", arch_name, 512);
    if (result == -1) {
        return -1;
    }
    arch_name[512] = 0;
    char context_length_name[600];
    memset(context_length_name, 0, sizeof(context_length_name));
    snprintf(context_length_name, 512, "%s.context_length", arch_name);

    char context_len_str[51];
    memset(context_len_str, 0, sizeof(context_len_str));

    result = llama_model_meta_val_str(model, context_length_name, context_len_str, 50);
    if (result == -1) {
        return -1;
    }

    char* endptr = 0;
    long context_len = strtol(context_len_str, &endptr, 10);
    if (!endptr || *endptr != 0) {
        return -1;
    }
    if (context_len > INT_MAX || context_len < 0) {
        return -1;
    }

    return (int) context_len;
}

llama_context* hs_llama_create_context(llama_model* model, int n_batch, int n_ctx, int n_threads, int n_threads_batch) {
    assert(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_batch = n_batch;
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads_batch;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    return ctx;
}

void hs_llama_free_context(llama_context* ctx) {
    if (ctx) {
        llama_free(ctx);
    }
}

int hs_llama_tokenize(llama_model* model, const char* text, int32_t** tokens, size_t* tokens_len) {
    assert(model);
    assert(text);
    assert(tokens);
    assert(tokens_len);

    int ntokens = ::llama_tokenize(model, text, strlen(text), 0, 0, false, false);
    if (ntokens == 0) {
        (*tokens) = 0;
        (*tokens_len) = 0;
        return 0;
    }
    ntokens = -ntokens;

    (*tokens) = (int32_t*) calloc(ntokens, sizeof(int32_t));
    if ((*tokens) == NULL) {
        return 1;
    }
    (*tokens_len) = ntokens;
    ::llama_tokenize(model, text, strlen(text), *tokens, *tokens_len, false, false);
    return 0;
}

void hs_free_tokens(int32_t* tokens) {
    free(tokens);
}

int32_t hs_bos_token(llama_context* context) {
    assert(context);
    return llama_token_bos(llama_get_model(context));
}

int32_t hs_eos_token(llama_context* context) {
    assert(context);
    return llama_token_eos(llama_get_model(context));
}

int hs_token_to_text(llama_model* model, int32_t token, char** text, size_t* text_len) {
    assert(model);
    assert(text);
    assert(text_len);

    int len = llama_token_to_piece(model, token, 0, 0);
    if (len == 0) {
        (*text) = 0;
        (*text_len) = 0;
        return 0;
    }
    len = -len;
    (*text) = (char*) calloc(1, len+1);
    if (!(*text)) {
        return 1;
    }
    (*text_len) = len;
    llama_token_to_piece(model, token, *text, *text_len);
    return 0;
}

void hs_free_text(char* text) {
    free(text);
}

typedef struct hs_batch
{
    int capacity;
    llama_batch batch;
} hs_batch;

hs_batch* hs_create_batch(int sz) {
    hs_batch* b = (hs_batch*) calloc(1, sizeof(hs_batch));
    if (!b) {
        return 0;
    }
    b->batch = llama_batch_init(sz, 0, 1);
    b->capacity = sz;
    b->batch.n_tokens = 0;
    memset(b->batch.token, 0, sz * sizeof(llama_token));
    memset(b->batch.pos, 0, sz * sizeof(llama_pos));
    for (int token_idx = 0; token_idx < sz; token_idx++) {
        b->batch.n_seq_id[token_idx] = 1;
        b->batch.seq_id[token_idx][0] = 0;
    }
    memset(b->batch.logits, 0, sz * sizeof(int8_t));
    return b;
}

void hs_set_batch_item(hs_batch* batch, int idx, int token, int pos, int seq_id, int logits)
{
    assert(batch);

    batch->batch.token[idx] = token;
    batch->batch.pos[idx] = pos;
    batch->batch.logits[idx] = logits;
    batch->batch.n_seq_id[idx] = 1;
    batch->batch.seq_id[idx][0] = seq_id;
}

void hs_set_batch_length(hs_batch* batch, int len)
{
    assert(batch);
    batch->batch.n_tokens = len;
}

void hs_free_batch(hs_batch* batch) {
    if (!batch) {
        return;
    }
    llama_batch_free(batch->batch);
}

int hs_batch_capacity(hs_batch* batch) {
    assert(batch);
    return batch->capacity;
}

int hs_batch_length(hs_batch* batch) {
    assert(batch);
    return batch->batch.n_tokens;
}

int hs_decode(llama_context* ctx, hs_batch* batch) {
    assert(ctx);
    assert(batch);
    return llama_decode(ctx, batch->batch);
}

int hs_get_vocab_size(llama_context* ctx) {
    assert(ctx);

    const llama_model* model = llama_get_model(ctx);
    return llama_n_vocab(model);
}

void hs_get_logits(llama_context* ctx, int idx, float* logits) {
    assert(ctx);
    assert(logits);

    float* l = llama_get_logits_ith(ctx, idx);
    memcpy(logits, l, sizeof(float) * hs_get_vocab_size(ctx));
}

int32_t hs_sample_mirostat(llama_context* ctx,
                           float* logits,
                           float* mu,
                           uint8_t* blacklist,
                           float tau,
                           float eta) {
    assert(ctx);
    assert(logits);
    assert(mu);
    // blacklist allowed to be NULL

    llama_token_data_array arr;
    memset(&arr, 0, sizeof(arr));
    int vocab_size = hs_get_vocab_size(ctx);
    arr.data = (llama_token_data*) calloc(vocab_size, sizeof(llama_token_data));
    if (!arr.data) {
        fprintf(stderr, "Failed to allocate memory for token data array\n");
        abort();
    }

    arr.size = 0;
    arr.sorted = false;

    int arr_cursor = 0;
    for (int i1 = 0; i1 < vocab_size; ++i1) {
        if (blacklist && blacklist[i1]) {
            continue;
        }
        arr.data[arr_cursor].id = i1;
        arr.data[arr_cursor].logit = logits[i1];
        arr.data[arr_cursor].p = 0.0;
        arr_cursor++;
        arr.size++;
    }

    if (arr_cursor == 0) {
        free(arr.data);
        // everything was blacklisted
        return -1;
    }

    int32_t result = llama_sample_token_mirostat_v2(ctx, &arr, tau, eta, mu);
    free(arr.data);

    return result;
}

void hs_remove_tokens(llama_context* ctx, int seq_id, int start, int end) {
    assert(ctx);
    llama_kv_cache_seq_rm(ctx, seq_id, start, end);
}

}
