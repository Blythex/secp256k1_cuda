#include <cuda_runtime.h>
#include <stdint.h>

#define p_value 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

typedef uint64_t ulong;

struct uint256_t {
    ulong values[4];
};

struct ECPoint {
    uint256_t x;
    uint256_t y;
};

__device__ inline uint256_t ZERO() { 
    uint256_t value = {0, 0, 0, 0};
    return value;
}

__device__ inline uint256_t ONE() { 
    uint256_t value = {1, 0, 0, 0};
    return value;
}

__device__ inline uint256_t TWO() { 
    uint256_t value = {2, 0, 0, 0};
    return value;
}

__device__ inline uint256_t THREE() { 
    uint256_t value = {3, 0, 0, 0};
    return value;
}

__device__ int cmp256(uint256_t a, uint256_t b) {
    for (int i = 3; i >= 0; i--) {
        if (a.values[i] > b.values[i]) return 1;
        if (a.values[i] < b.values[i]) return -1;
    }
    return 0;
}

__device__ void mul_extended(ulong a, ulong b, ulong *high, ulong *low) {
    ulong a_lo = a & 0xFFFFFFFF;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF;
    ulong b_hi = b >> 32;

    ulong lo_lo = a_lo * b_lo;
    ulong hi_lo = a_hi * b_lo;
    ulong lo_hi = a_lo * b_hi;
    ulong hi_hi = a_hi * b_hi;

    ulong cross = hi_lo + lo_hi;
    *low = lo_lo + (cross << 32);
    *high = hi_hi + (cross >> 32);
}

__device__ uint256_t mod_p(uint256_t a) {
    return mod256(a, P_VALUE);
}



__device__ uint256_t add256(uint256_t a, uint256_t b) {
    uint256_t result;
    ulong carry = 0;
    for (int i = 0; i < 4; i++) {
        result.values[i] = a.values[i] + b.values[i] + carry;
        carry = (result.values[i] < a.values[i] || (carry && result.values[i] == a.values[i])) ? 1 : 0;
    }
    return result;
}

__device__ uint256_t sub256(uint256_t a, uint256_t b) {
    uint256_t result;
    ulong borrow = 0;
    for (int i = 0; i < 4; i++) {
        result.values[i] = a.values[i] - b.values[i] - borrow;
        borrow = (a.values[i] < b.values[i] || (borrow && a.values[i] == b.values[i])) ? 1 : 0;
    }
    return result;
}

__device__ uint256_t mul256(uint256_t a, uint256_t b) {
    uint256_t result = ZERO();
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong high, low;
            mul_extended(a.values[i], b.values[j], &high, &low);
            ulong sum = result.values[i + j] + low + carry;
            carry = high + (sum < low);
            result.values[i + j] = sum;
        }
        int k = i + 1;
        while (carry && k < 4) {
            result.values[k] += carry;
            carry = result.values[k] < carry;
            k++;
        }
    }
    return result;
}

__device__ uint256_t mod256(uint256_t a, uint256_t b) {
    while (cmp256(a, b) >= 0) {
        a = sub256(a, b);
    }
    return a;
}

__device__ uint256_t div256(uint256_t a, uint256_t b) {
    uint256_t quotient = ZERO();
    while (cmp256(a, b) >= 0) {
        a = sub256(a, b);
        quotient = add256(quotient, ONE());
    }
    return quotient;
}

__device__ uint256_t uint256_from_ulong(ulong val) {
    uint256_t result;
    result.values[0] = val;
    for (int i = 1; i < 4; i++) {
        result.values[i] = 0;
    }
    return result;
}

__device__ uint256_t mod_pow256(uint256_t base, uint256_t exp, uint256_t mod) {
    uint256_t result = {1, 0, 0, 0};
    while (cmp256(exp, ZERO()) > 0) {
        if (exp.values[0] & 1) {
            result = mul256(result, base);
            result = mod256(result, mod);
        }
        base = mul256(base, base);
        base = mod256(base, mod);
        ulong carry = 0;
        for (int i = 3; i > 0; i--) {
            carry = (exp.values[i] & 1) << 63;
            exp.values[i] >>= 1;
            exp.values[i] |= carry;
        }
        exp.values[0] >>= 1;
    }
    return result;
}

__device__ uint256_t inverse256(uint256_t a) {
    uint256_t p_minus_2 = {
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFEFFFFFC2D
    };
    return mod_pow256(a, p_minus_2, P_VALUE);
}

__device__ ECPoint EC_Double(ECPoint P) {
    if (cmp256(P.y, ZERO()) == 0) {
        return {ZERO(), ZERO()};
    }

    uint256_t three = THREE();
    uint256_t two = TWO();

    uint256_t lambda = mod_p(mul256(three, mul256(P.x, P.x)));
    uint256_t inv_2y = inverse256(mul256(two, P.y));
    lambda = mod_p(mul256(lambda, inv_2y));

    uint256_t x3 = mod_p(sub256(mul256(lambda, lambda), add256(P.x, P.x)));
    uint256_t y3 = mod_p(sub256(mul256(lambda, sub256(P.x, x3)), P.y));

    return {x3, y3};
}

__device__ ECPoint EC_Add(ECPoint P, ECPoint Q) {
    // Fall: P ist der Unendlichkeitspunkt
    if (cmp256(P.x, ZERO()) == 0 && cmp256(P.y, ZERO()) == 0) {
        return Q;
    }

    // Fall: Q ist der Unendlichkeitspunkt
    if (cmp256(Q.x, ZERO()) == 0 && cmp256(Q.y, ZERO()) == 0) {
        return P;
    }

    // Fall: P und Q sind identisch
    if (cmp256(P.x, Q.x) == 0 && cmp256(P.y, Q.y) == 0) {
        return EC_Double(P);
    }

    // Fall: P und Q haben dieselben x-Koordinaten, aber unterschiedliche y-Koordinaten
    if (cmp256(P.x, Q.x) == 0) {
        return {ZERO(), ZERO()};  // Ergebnis ist der Unendlichkeitspunkt
    }

    uint256_t s, x3, y3;

    uint256_t diff_y = sub256(Q.y, P.y);
    uint256_t diff_x = sub256(Q.x, P.x);
    uint256_t inv_diff_x = inverse256(diff_x);
    s = mod_p(mul256(diff_y, inv_diff_x));
    x3 = mod_p(sub256(sub256(mul256(s, s), P.x), Q.x));
    y3 = mod_p(sub256(mul256(s, sub256(P.x, x3)), P.y));

    ECPoint R = {x3, y3};
    return R;
}


__device__ int is_point_on_curve(ECPoint P) {
    uint256_t lhs = mod_p(mul256(P.y, P.y));
    uint256_t rhs = mod_p(add256(add256(mul256(P.x, mul256(P.x, P.x)), mul256(SEVEN(), P.x)), SEVEN())); // y^2 = x^3 + 7

    return cmp256(lhs, rhs) == 0;
}


__device__ ECPoint EC_Multiply(uint256_t scalar, ECPoint P) {
    ECPoint result = {ZERO(), ZERO()};
    ECPoint addend = P;

    for (int i = 3; i >= 0; i--) {
        ulong current = scalar.values[i];
        for (int j = 0; j < 64; j++) {
            if (current & 1) {
                result = EC_Add(result, addend);
            }
            addend = EC_Add(addend, addend);
            current >>= 1;
        }
    }

    return result;
}


__global__ void scalar_multiplication(uint256_t* private_keys, unsigned char* public_keys, int num_keys) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;

    uint256_t private_key = private_keys[gid];

    ECPoint G = { 
        {0x79BE667EF9DCBBAC, 0x55A06295CE870B07, 0x029BFCDB2DCE28D9, 0x59F2815B16F81798}, 
        {0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8, 0xFD17B448A6855419, 0x9C47D08FFB10D4B8}
    }; 

    ECPoint result = EC_Multiply(private_key, G);

    unsigned char public_key[65];
    public_key[0] = 0x04;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 8; i++) {
            public_key[j * 8 + i + 1] = (unsigned char)(result.x.values[j] >> (8 * (7 - i)));
            public_key[j * 8 + i + 33] = (unsigned char)(result.y.values[j] >> (8 * (7 - i)));
        }
    }

    for (int i = 0; i < 65; i++) {
        public_keys[gid * 65 + i] = public_key[i];
    }
}

__global__ void count_generated_keys(int *keys_generated) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&keys_generated[gid], 1);
}
