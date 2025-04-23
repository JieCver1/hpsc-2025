#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }
  for(int i=0; i<N; i++) {
    /*for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }*/
	 //x[i],y[i]
	 __m512 xivec = _mm512_set1_ps(x[i]);
	 __m512 yivec = _mm512_set1_ps(y[i]);
	 
	 //x[j],x[j]
	 __m512 xjvec = _mm512_load_ps(x);
	 __m512 yjvec = _mm512_load_ps(y);
	 
	 //m[j]
	 __m512 mjvec = _mm512_load_ps(m);

	 //rx,ry
	 __m512 rxvec = _mm512_sub_ps(xivec,xjvec);
	 __m512 ryvec = _mm512_sub_ps(yivec,yjvec);

         //(rx * rx + ry * ry)
	 __m512 rxQ = _mm512_mul_ps(rxvec,rxvec);
	 __m512 ryQ = _mm512_mul_ps(ryvec,ryvec);
	 __m512 sum = _mm512_add_ps(rxQ,ryQ);

	 //float r = std::sqrt(rx * rx + ry * ry);
	 __m512 rvec = _mm512_rsqrt14_ps(sum);
	
	 //r*r*r
         __m512 prodR = _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec));
         
	 //rx * m[j] / (r * r * r); and ry * m[j] / (r * r * r); 
	 __m512 prodMR = _mm512_mul_ps(mjvec, prodR);

         __m512 fxivec = _mm512_mul_ps(rxvec, prodMR);
         __m512 fyivec = _mm512_mul_ps(ryvec, prodMR);
	 
	 __m512 fxTmp = _mm512_load_ps(fx);
         __m512 fyTmp = _mm512_load_ps(fy);
         __m512 fxi = _mm512_setzero_ps();
         __m512 fyi = _mm512_setzero_ps();

	 __m512 jvec = _mm512_load_ps(j);
         __m512 ivec = _mm512_set1_ps(i);
	 __mmask16 mask = _mm512_cmp_ps_mask(ivec, jvec, _MM_CMPINT_NE);

	 fxi = _mm512_mask_blend_ps(mask, fxi, fxivec);
         fyi = _mm512_mask_blend_ps(mask, fyi, fyivec);
         fx[i] -= _mm512_reduce_add_ps(fxi);
         fy[i] -= _mm512_reduce_add_ps(fyi);
    //}
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
