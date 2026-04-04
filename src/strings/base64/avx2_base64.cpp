#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

extern "C" {

// =======================================================================
// 1. STRATEGY 2: THE BMI2 / AVX2 DESPACER
// =======================================================================

static const int8_t lut_lo_arr[32] __attribute__((aligned(32))) = {
    0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
    0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A,
    0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
    0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A
};

static const int8_t lut_hi_arr[32] __attribute__((aligned(32))) = {
    0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08,
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10
};

static const int8_t lut_roll_arr[32] __attribute__((aligned(32))) = {
    0,  16,  19,   4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  16,  19,   4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0
};

// Scalar Decoding Table for the Tail
static const int8_t DT[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63, 52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, 15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40, 41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1
};

static inline bool fromascii(__m256i str, __m256i* out) {
    __m256i mask_2F = _mm256_set1_epi8(0x2F);
    __m256i mask_0F = _mm256_set1_epi8(0x0F);
    
    __m256i v_lut_lo = _mm256_loadu_si256((const __m256i*)lut_lo_arr);
    __m256i v_lut_hi = _mm256_loadu_si256((const __m256i*)lut_hi_arr);
    __m256i v_lut_roll = _mm256_loadu_si256((const __m256i*)lut_roll_arr);

    __m256i hi_nibbles = _mm256_srli_epi32(str, 4);
    __m256i lo_nibbles = _mm256_and_si256(str, mask_0F); 
    
    __m256i lo = _mm256_shuffle_epi8(v_lut_lo, lo_nibbles);
    __m256i eq_2F = _mm256_cmpeq_epi8(str, mask_2F);
    
    hi_nibbles = _mm256_and_si256(hi_nibbles, mask_0F);
    __m256i hi = _mm256_shuffle_epi8(v_lut_hi, hi_nibbles);
    __m256i roll = _mm256_shuffle_epi8(v_lut_roll, _mm256_add_epi8(eq_2F, hi_nibbles));
                        
    if (!_mm256_testz_si256(lo, hi)) return false; 
    
    *out = _mm256_add_epi8(str, roll);
    return true;
}

static inline __m256i dec_reshuffle(__m256i in) {
    __m256i merge_ab_and_bc = _mm256_maddubs_epi16(in, _mm256_set1_epi32(0x01400140));
    __m256i out = _mm256_madd_epi16(merge_ab_and_bc, _mm256_set1_epi32(0x00011000));
                      
    out = _mm256_shuffle_epi8(out, _mm256_setr_epi8(
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1,
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1
    ));
    
    return _mm256_permutevar8x32_epi32(out, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, -1, -1));
}

// =======================================================================
// 2. FUSED AVX2 DECODER (ONE-PASS)
// =======================================================================

__attribute__((target("avx2,bmi2")))
size_t decode_avx2_chunked(const int8_t* src, size_t srclen, char* dst, size_t dstlen, bool* error_flag) {
    size_t i = 0;
    size_t out_idx = 0;
    *error_flag = false;

    while (srclen > 0 && (uint8_t)src[srclen - 1] <= 32) {
        srclen--;
    }

    // L1-Cache Spill Buffer
    uint8_t q[128   ] __attribute__((aligned(32)));
    int q_len = 0;
    const __m256i space_thresh = _mm256_set1_epi8(32);

    // -------------------------------------------------------------------
    // 1. FAT UNROLLED LOOP (128-byte stride)
    // -------------------------------------------------------------------
    while (i + 128 + 32 <= srclen) {
        __m256i c0 = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i c1 = _mm256_loadu_si256((const __m256i*)(src + i + 32));
        __m256i c2 = _mm256_loadu_si256((const __m256i*)(src + i + 64));
        __m256i c3 = _mm256_loadu_si256((const __m256i*)(src + i + 96));

        __m256i m0 = _mm256_cmpeq_epi8(_mm256_max_epu8(c0, space_thresh), space_thresh);
        __m256i m1 = _mm256_cmpeq_epi8(_mm256_max_epu8(c1, space_thresh), space_thresh);
        __m256i m2 = _mm256_cmpeq_epi8(_mm256_max_epu8(c2, space_thresh), space_thresh);
        __m256i m3 = _mm256_cmpeq_epi8(_mm256_max_epu8(c3, space_thresh), space_thresh);

        __m256i any_space = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));

        if (_mm256_testz_si256(any_space, any_space) && q_len == 0) {
            // FAST PATH: Perfect 128-bytes, no spaces, empty queue.
            __m256i dec0, dec1, dec2, dec3;
            if (!fromascii(c0, &dec0)) { *error_flag = true; return out_idx; }
            if (!fromascii(c1, &dec1)) { *error_flag = true; return out_idx; }
            if (!fromascii(c2, &dec2)) { *error_flag = true; return out_idx; }
            if (!fromascii(c3, &dec3)) { *error_flag = true; return out_idx; }

            __m256i p0 = dec_reshuffle(dec0);
            __m256i p1 = dec_reshuffle(dec1);
            __m256i p2 = dec_reshuffle(dec2);
            __m256i p3 = dec_reshuffle(dec3);

            _mm256_storeu_si256((__m256i*)(dst + out_idx), p0);
            _mm256_storeu_si256((__m256i*)(dst + out_idx + 24), p1);
            _mm256_storeu_si256((__m256i*)(dst + out_idx + 48), p2);
            _mm_storeu_si128((__m128i*)(dst + out_idx + 72), _mm256_castsi256_si128(p3));
            _mm_storel_epi64((__m128i*)(dst + out_idx + 88), _mm256_extracti128_si256(p3, 1));

            i += 128;
            out_idx += 96;
        } else {
            // BUFFER PATH: Handle block with spaces or carry-over carefully
            for (int k = 0; k < 4; ++k) {
                __m256i chunk = _mm256_loadu_si256((const __m256i*)(src + i));
                __m256i le_32 = _mm256_cmpeq_epi8(_mm256_max_epu8(chunk, space_thresh), space_thresh);
                uint32_t mask32 = ~_mm256_movemask_epi8(le_32);

                if (mask32 == 0xFFFFFFFF && q_len == 0) {
                    __m256i dec;
                    if (!fromascii(chunk, &dec)) { *error_flag = true; return out_idx; }
                    __m256i packed = dec_reshuffle(dec);
                    _mm_storeu_si128((__m128i*)(dst + out_idx), _mm256_castsi256_si128(packed));
                    _mm_storel_epi64((__m128i*)(dst + out_idx + 16), _mm256_extracti128_si256(packed, 1));
                    out_idx += 24;
                } else {
                    for (int j = 0; j < 4; ++j) {
                        uint64_t word;
                        memcpy(&word, src + i + j * 8, 8);
                        uint8_t m8 = (mask32 >> (j * 8)) & 0xFF;
                        uint64_t pext_mask = _pdep_u64(m8, 0x0101010101010101ULL) * 0xFF;
                        uint64_t packed = _pext_u64(word, pext_mask);
                        memcpy(q + q_len, &packed, 8);
                        q_len += __builtin_popcount(m8); 
                    }

                    while (q_len >= 32) {
                        __m256i q_chunk = _mm256_loadu_si256((const __m256i*)q);
                        __m256i dec;
                        if (!fromascii(q_chunk, &dec)) { *error_flag = true; return out_idx; }
                        __m256i packed = dec_reshuffle(dec);
                        
                        _mm_storeu_si128((__m128i*)(dst + out_idx), _mm256_castsi256_si128(packed));
                        _mm_storel_epi64((__m128i*)(dst + out_idx + 16), _mm256_extracti128_si256(packed, 1));
                        out_idx += 24;

                        _mm256_storeu_si256((__m256i*)q, _mm256_loadu_si256((const __m256i*)(q + 32)));
                        q_len -= 32;
                    }
                }
                i += 32;
            }
        }
    }

    // -------------------------------------------------------------------
    // 2. REMAINDER LOOP (32-byte stride)
    // -------------------------------------------------------------------
    while (i + 32 + 32 <= srclen) {
        __m256i chunk = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i le_32 = _mm256_cmpeq_epi8(_mm256_max_epu8(chunk, space_thresh), space_thresh);
        uint32_t mask32 = ~_mm256_movemask_epi8(le_32);

        if (mask32 == 0xFFFFFFFF && q_len == 0) {
            __m256i dec;
            if (!fromascii(chunk, &dec)) { *error_flag = true; return out_idx; }
            __m256i packed = dec_reshuffle(dec);
            _mm_storeu_si128((__m128i*)(dst + out_idx), _mm256_castsi256_si128(packed));
            _mm_storel_epi64((__m128i*)(dst + out_idx + 16), _mm256_extracti128_si256(packed, 1));
            out_idx += 24;
        } else {
            for (int j = 0; j < 4; ++j) {
                uint64_t word;
                memcpy(&word, src + i + j * 8, 8);
                uint8_t m8 = (mask32 >> (j * 8)) & 0xFF;
                uint64_t pext_mask = _pdep_u64(m8, 0x0101010101010101ULL) * 0xFF;
                uint64_t packed = _pext_u64(word, pext_mask);
                memcpy(q + q_len, &packed, 8);
                q_len += __builtin_popcount(m8); 
            }

            while (q_len >= 32) {
                __m256i q_chunk = _mm256_loadu_si256((const __m256i*)q);
                __m256i dec;
                if (!fromascii(q_chunk, &dec)) { *error_flag = true; return out_idx; }
                __m256i packed = dec_reshuffle(dec);
                _mm_storeu_si128((__m128i*)(dst + out_idx), _mm256_castsi256_si128(packed));
                _mm_storel_epi64((__m128i*)(dst + out_idx + 16), _mm256_extracti128_si256(packed, 1));
                out_idx += 24;

                _mm256_storeu_si256((__m256i*)q, _mm256_loadu_si256((const __m256i*)(q + 32)));
                q_len -= 32;
            }
        }
        i += 32;
    }

    // -------------------------------------------------------------------
    // 3. SCALAR TAIL (Drain the queue + handle final source bytes)
    // -------------------------------------------------------------------
    char tail_buffer[128]; 
    int tail_len = 0;
    
    // Dump remaining queue
    for (int k = 0; k < q_len; k++) {
        tail_buffer[tail_len++] = q[k];
    }
    
    // Dump remaining source bytes (ignoring spaces/control characters)
    for (; i < srclen; i++) {
        if ((uint8_t)src[i] > 32) {
            tail_buffer[tail_len++] = src[i];
        }
    }

    // STRICT COMPLIANCE: Valid Base64 (excluding spaces) must be a multiple of 4.
    // If it is not, the string was truncated or malformed.
    if (tail_len % 4 != 0) {
        *error_flag = true;
        return 0; // Return 0 on fatal error to ensure Fortran aborts
    }

    // Decode tail buffer using scalar table
    for (int k = 0; k < tail_len; k += 4) {
        // STRICT COMPLIANCE: Reject non-ASCII high-bit characters safely
        if ((uint8_t)tail_buffer[k] > 127 || (uint8_t)tail_buffer[k+1] > 127 ||
            (uint8_t)tail_buffer[k+2] > 127 || (uint8_t)tail_buffer[k+3] > 127) {
            *error_flag = true;
            return 0; 
        }

        int8_t v0 = DT[(uint8_t)tail_buffer[k]];
        int8_t v1 = DT[(uint8_t)tail_buffer[k+1]];
        int8_t v2 = DT[(uint8_t)tail_buffer[k+2]];
        int8_t v3 = DT[(uint8_t)tail_buffer[k+3]];

        // The first two characters of a 4-byte block MUST be valid data
        if (v0 < 0 || v1 < 0) {
            *error_flag = true;
            return 0; 
        }

        dst[out_idx++] = (char)((v0 << 2) | (v1 >> 4));

        // The third character can be data, or it MUST be '='
        if (v2 >= 0) {
            // --- BYTE 2 CHECK ---
            if (out_idx >= dstlen) { *error_flag = true; return 0; }
            dst[out_idx++] = (char)((v1 << 4) | (v2 >> 2));
            
            if (v3 >= 0) {
                // --- BYTE 3 CHECK ---
                if (out_idx >= dstlen) { *error_flag = true; return 0; }
                dst[out_idx++] = (char)((v2 << 6) | v3);
            } else if (tail_buffer[k+3] != '=') {
                *error_flag = true; return 0;
            }
        } else if (tail_buffer[k+2] != '=' || tail_buffer[k+3] != '=') {
            *error_flag = true; return 0;
        }
    }

    return out_idx;
}

static const char* base64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void base64_encode_scalar(const uint8_t* src, size_t len, uint8_t* dst) {
    size_t i = 0, j = 0;
    while (i + 2 < len) {
        uint32_t octet_a = src[i];
        uint32_t octet_b = src[i+1];
        uint32_t octet_c = src[i+2];
        i += 3;
        uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;
        dst[j++] = base64_table[(triple >> 18) & 0x3F];
        dst[j++] = base64_table[(triple >> 12) & 0x3F];
        dst[j++] = base64_table[(triple >> 6) & 0x3F];
        dst[j++] = base64_table[triple & 0x3F];
    }
    
    if (i < len) {
        size_t rem = len - i;
        uint32_t octet_a = src[i];
        uint32_t octet_b = (rem > 1) ? src[i+1] : 0;
        uint32_t triple = (octet_a << 16) | (octet_b << 8);
        dst[j++] = base64_table[(triple >> 18) & 0x3F];
        dst[j++] = base64_table[(triple >> 12) & 0x3F];
        dst[j++] = (rem > 1) ? base64_table[(triple >> 6) & 0x3F] : '=';
        dst[j++] = '=';
    }
}

__attribute__((optimize("O3", "unroll-loops", "no-tree-loop-distribute-patterns"), target("avx2"), noinline))
void base64_encode_avx2(const uint8_t* src, size_t len, uint8_t* dst) {
    size_t i = 0, j = 0;
    
    // 1. DYNAMIC ALIGNMENT PROLOGUE
    // Attempts to align dst to 32-bytes for optimal bus transfer speeds.
    size_t unaligned_bytes = (uintptr_t)dst % 32;
    if (len >= 32 && unaligned_bytes != 0) {
        size_t b_align = 32 - unaligned_bytes;
        if (b_align % 4 == 0) {
            size_t in_p = (b_align / 4) * 3;
            base64_encode_scalar(src, in_p, dst);
            i += in_p; j += b_align;
        }
    }

    // 2. CONSTANTS
    const __m256i shuf = _mm256_setr_epi8(1,0,2,1, 4,3,5,4, 7,6,8,7, 10,9,11,10, 1,0,2,1, 4,3,5,4, 7,6,8,7, 10,9,11,10);
    const __m256i perm = _mm256_setr_epi32(0,1,2,0, 3,4,5,0);
    const __m256i t0_m = _mm256_set1_epi32(0x0FC0FC00);
    const __m256i t1_m = _mm256_set1_epi32(0x04000040);
    const __m256i t2_m = _mm256_set1_epi32(0x003F03F0);
    const __m256i t3_m = _mm256_set1_epi32(0x01000010);
    const __m256i r_sb = _mm256_set1_epi8(51);
    const __m256i r_cp = _mm256_set1_epi8(26);
    const __m256i l_os = _mm256_set1_epi8('A');
    const __m256i lut  = _mm256_setr_epi8('a'-26,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'+'-62,'/'-63,'A','A','A','a'-26,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'0'-52,'+'-62,'/'-63,'A','A','A');

    // 3. THE "BLACK BOX" UNROLLED LOOP
    // STRICT BOUNDS: 8 steps. The last step starts at i + 168 and loads 32 bytes.
    // 168 + 32 = 200. This guarantees we NEVER over-read the src array.
    while (i + 200 <= len) {
        const uint8_t* s_ptr = src + i;
        uint8_t* d_ptr = dst + j;

        #define ENCODE_STEP_UNROLLED { \
            __m256i in = _mm256_loadu_si256((const __m256i*)s_ptr); \
            in = _mm256_permutevar8x32_epi32(in, perm); \
            in = _mm256_shuffle_epi8(in, shuf); \
            __m256i ind = _mm256_or_si256(_mm256_mulhi_epu16(_mm256_and_si256(in, t0_m), t1_m), _mm256_mullo_epi16(_mm256_and_si256(in, t2_m), t3_m)); \
            in = _mm256_add_epi8(ind, _mm256_blendv_epi8(_mm256_shuffle_epi8(lut, _mm256_subs_epu8(ind, r_sb)), l_os, _mm256_cmpgt_epi8(r_cp, ind))); \
            _mm256_storeu_si256((__m256i*)d_ptr, in); \
            s_ptr += 24; d_ptr += 32; \
            asm volatile("" : "+r"(s_ptr), "+r"(d_ptr) :: "memory"); \
        }
        // NOTE: Changed store_si256 to storeu_si256 above to prevent Fortran slice crashes.

        ENCODE_STEP_UNROLLED; ENCODE_STEP_UNROLLED;
        ENCODE_STEP_UNROLLED; ENCODE_STEP_UNROLLED;
        ENCODE_STEP_UNROLLED; ENCODE_STEP_UNROLLED;
        ENCODE_STEP_UNROLLED; ENCODE_STEP_UNROLLED;

        i += 192;
        j += 256;
    }

    // 4. REMAINDER LOOP
    // STRICT BOUNDS: i + 32 guarantees the loadu_si256 won't touch unallocated memory.
    while (i + 32 <= len) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
        v = _mm256_permutevar8x32_epi32(v, perm);
        v = _mm256_shuffle_epi8(v, shuf);
        __m256i ind = _mm256_or_si256(_mm256_mulhi_epu16(_mm256_and_si256(v, t0_m), t1_m), _mm256_mullo_epi16(_mm256_and_si256(v, t2_m), t3_m));
        v = _mm256_add_epi8(ind, _mm256_blendv_epi8(_mm256_shuffle_epi8(lut, _mm256_subs_epu8(ind, r_sb)), l_os, _mm256_cmpgt_epi8(r_cp, ind)));
        _mm256_storeu_si256((__m256i*)(dst + j), v);
        i += 24; j += 32;
    }

    // 5. SCALAR TAIL
    // Handles the remaining 0 to 31 bytes cleanly.
    if (i < len) base64_encode_scalar(src + i, len - i, dst + j);
}

} // extern "C"