#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  
  std::vector<int>tmp(range,0);

  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);

#pragma omp parallel
  for (int j=1; j<range; j<<=1){
#pragma omp for
   for (int i=0; i<range; i++)
	tmp[i] = offset[i];
#pragma omp for
   for (int i=j; i<range; i++){
	offset[i] += tmp[i-j] + bucket[i-j];
   }
  }
#pragma omp for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
