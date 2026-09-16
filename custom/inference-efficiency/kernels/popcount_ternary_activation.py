"""W2A2 variant: sign plus nonzero planes for both dot-product operands.

Activation words hold sign in low 32 bits and nonzero in high 32 bits.
The fixed activation quantizer maps x>=.5 to +1, x<=-.5 to -1, else zero.
"""
import functools

from . import popcount
from .cuda_gemv import _preload_nvrtc


SOURCE = popcount.SOURCE.replace('const u32* A', 'const unsigned long long* A')
SOURCE = SOURCE.replace('u32* A, int M', 'unsigned long long* A, int M')
SOURCE = SOURCE.replace(
    'u32 v = __ballot_sync(0xffffffffu, positive);',
    '''u32 v = __ballot_sync(0xffffffffu, positive);
    bool nonzero = m < M && k < K &&
        (h2f(X[(long long)m*K+k]) >= .5f || h2f(X[(long long)m*K+k]) <= -.5f);
    u32 z = __ballot_sync(0xffffffffu, nonzero);''')
SOURCE = SOURCE.replace('A[word] = v;', 'A[word] = ((unsigned long long)z << 32) | v;')
SOURCE = SOURCE.replace(
    'sum += contribution<TERNARY>',
    'if (TERNARY) nz &= (u32)(A[(long long)m*KW+q] >> 32);\n        sum += contribution<TERNARY>')


@functools.lru_cache(maxsize=1)
def load():
    _preload_nvrtc()
    from torch.cuda import _compile_kernel
    names = ['pack_sign'] + [f'ternary_{layout}{suffix}'
                            for layout in ('warp', 'tile')
                            for suffix in ('', '_k1024', '_k4096')]
    return {name: _compile_kernel(SOURCE, name) for name in names}


def pack(x, out):
    m, k = x.shape
    kw = (k + 31) // 32
    load()['pack_sign'](grid=((m * kw + 7) // 8, 1, 1), block=(256, 1, 1),
                        args=[x, out, m, k, kw])


class BitWeight(popcount.BitWeight):
    def linear(self, a, out, variant='warp', bias=None):
        if self.bits != 2:
            raise ValueError('This experiment is W2A2 only')
        m = a.shape[0]
        tile = variant == 'tile'
        sign, nz = (self.sign_t, self.nonzero_t) if tile else (self.sign, self.nonzero)
        name = 'ternary_' + variant
        if self.k in (1024, 4096):
            name += f'_k{self.k}'
        load()[name](grid=((self.n + (31 if tile else 3)) // (32 if tile else 4),
                           (m+3)//4 if tile else m, 1), block=(128, 1, 1),
                     args=[a, sign, nz, self.scales, bias if bias is not None else out,
                           out, m, self.n, self.k, a.shape[1], int(bias is not None)])
        return out
