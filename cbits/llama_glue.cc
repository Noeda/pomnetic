#include "llama.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef _WIN32
#include <signal.h>
#endif

extern "C" {

int hs_llama_kill_pid(long pid) {
    return kill(pid, SIGKILL);
}

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

int32_t hs_bos_token_model(llama_model* model) {
    assert(model);
    return llama_token_bos(model);
}

int32_t hs_eos_token_model(llama_model* model) {
    assert(model);
    return llama_token_eos(model);
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

    // stores logits; if requested.
    // always same size as capacity but some entries can be null, or
    // allocated/freed on the fly when the batch is used again.
    //
    // not to be confused with llama_batch.logits which just indicates if
    // logits will be calculated.
    float** logits;
    // stores lengths of logits
    size_t* logits_len;

    llama_batch batch;
} hs_batch;

hs_batch* hs_create_batch(int sz) {
    hs_batch* b = (hs_batch*) calloc(1, sizeof(hs_batch));
    if (!b) {
        return 0;
    }
    b->logits = (float**) calloc(sz, sizeof(float*));
    if (!b->logits) {
        free(b);
        return 0;
    }
    b->logits_len = (size_t*) calloc(sz, sizeof(size_t));
    if (!b->logits_len) {
        free(b->logits);
        free(b);
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
    if (batch->logits) {
        for (int i = 0; i < batch->capacity; i++) {
            free(batch->logits[i]);
        }
        free(batch->logits);
    }
    free(batch->logits_len);
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

int hs_get_vocab_size_model(llama_model* model) {
    assert(model);
    return llama_n_vocab(model);
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

int hs_decode(llama_context* ctx, hs_batch* batch) {
    assert(ctx);
    assert(batch);

    for (int i = 0; i < batch->batch.n_tokens; ++i) {
        if (batch->batch.logits[i]) {
            if (!batch->logits[i]) {
                batch->logits[i] = (float*) calloc(hs_get_vocab_size(ctx), sizeof(float));
                if (!batch->logits[i]) {
                    // FIXME: bad magic number
                    return 81273997;
                }
            } else {
                memset(batch->logits[i], 0, hs_get_vocab_size(ctx) * sizeof(float));
            }
            batch->logits_len[i] = hs_get_vocab_size(ctx);
        } else {
            free(batch->logits[i]);
            batch->logits[i] = 0;
            batch->logits_len[i] = 0;
        }
    }

    uint32_t cbatch_size = llama_n_batch(ctx);

    // do we need to split the batch?
    if (batch->batch.n_tokens > cbatch_size) {
        uint32_t tokens_left = batch->batch.n_tokens;
        uint32_t cursor = 0;
        llama_batch subbatch;
        subbatch = llama_batch_init(cbatch_size, 0, 1);

        while (tokens_left > 0) {
            subbatch.n_tokens = (tokens_left > cbatch_size) ? cbatch_size : tokens_left;
            memcpy(subbatch.token, &batch->batch.token[cursor], subbatch.n_tokens * sizeof(llama_token));
            memcpy(subbatch.n_seq_id, &batch->batch.n_seq_id[cursor], subbatch.n_tokens * sizeof(int32_t));
            for (int token_idx = 0; token_idx < subbatch.n_tokens; token_idx++) {
                subbatch.seq_id[token_idx][0] = batch->batch.seq_id[cursor + token_idx][0];
            }
            memcpy(subbatch.pos, &batch->batch.pos[cursor], subbatch.n_tokens * sizeof(llama_pos));
            memcpy(subbatch.logits, &batch->batch.logits[cursor], subbatch.n_tokens * sizeof(int8_t));

            int result = llama_decode(ctx, subbatch);
            if (result != 0) {
                llama_batch_free(subbatch);
                return result;
            }
            for (int token_idx = 0; token_idx < subbatch.n_tokens; token_idx++) {
                if (subbatch.logits[token_idx]) {
                    assert(batch->logits[cursor + token_idx]);
                    hs_get_logits(ctx, token_idx, batch->logits[cursor + token_idx]);
                }
            }

            memcpy(&batch->batch.token[cursor], subbatch.token, subbatch.n_tokens * sizeof(llama_token));
            memcpy(&batch->batch.n_seq_id[cursor], subbatch.n_seq_id, subbatch.n_tokens * sizeof(int32_t));
            for (int token_idx = 0; token_idx < subbatch.n_tokens; token_idx++) {
                batch->batch.seq_id[cursor + token_idx][0] = subbatch.seq_id[token_idx][0];
            }
            memcpy(&batch->batch.pos[cursor], subbatch.pos, subbatch.n_tokens * sizeof(llama_pos));
            memcpy(&batch->batch.logits[cursor], subbatch.logits, subbatch.n_tokens * sizeof(int8_t));

            tokens_left -= subbatch.n_tokens;
            cursor += subbatch.n_tokens;
        }

        llama_batch_free(subbatch);
        return 0;
    } else {
        int result = llama_decode(ctx, batch->batch);
        if (result != 0) {
            return result;
        }

        for (int token_idx = 0; token_idx < batch->batch.n_tokens; token_idx++) {
            if (batch->batch.logits[token_idx]) {
                assert(batch->logits[token_idx]);
                hs_get_logits(ctx, token_idx, batch->logits[token_idx]);
            }
        }

        return 0;
    }
}

int hs_batch_has_logits(hs_batch* batch, int idx) {
    assert(batch);
    if (idx < 0 || idx >= batch->capacity) {
        return 0;
    }
    return batch->batch.logits[idx];
}

void hs_get_logits_from_hs_batch(hs_batch* batch, int idx, float* logits) {
    assert(batch);
    assert(logits);
    assert(batch->logits);
    assert(batch->logits[idx]);

    memcpy(logits, batch->logits[idx], sizeof(float) * batch->logits_len[idx]);
}

size_t hs_get_logits_len_from_hs_batch(hs_batch* batch, int idx) {
    assert(batch);
    assert(batch->logits_len);
    return batch->logits_len[idx];
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
