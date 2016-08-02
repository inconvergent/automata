#define THREADS _THREADS_
#define PIF 3.141592654f

__device__ void _increment_hit_count(
    const unsigned int grid_size,
    const unsigned int i,
    const unsigned int j,
    const bool *grid,
    const float *massx,
    const float *massy,
    const int *neigh,
    int *hits
    ){
  const int k = i*grid_size+j;

  const float one = 1.0f/(float)grid_size;
  float x = (float)i*one+massx[k]*one;
  float y = (float)j*one+massy[k]*one;
  unsigned int ii = (unsigned int)round(x*grid_size);
  unsigned int jj = (unsigned int)round(y*grid_size);

  /*float a = atan2(massy[k], massx[k]);*/
  /*a += PIF;*/
  /*a /= (PIF*0.125f);*/
  /*const int w = (int)floor(a);*/
  /*unsigned int ii = i;*/
  /*unsigned int jj = j;*/
  /**/
  /*if (w<1){*/
  /*  ii -= 1;*/
  /*}else if(w<3){*/
  /*  ii -= 1;*/
  /*  jj -= 1;*/
  /*}else if(w<5){*/
  /*  jj -= 1;*/
  /*}else if(w<7){*/
  /*  ii += 1;*/
  /*  jj -= 1;*/
  /*}else if(w<9){*/
  /*  ii += 1;*/
  /*}else if(w<11){*/
  /*  ii += 1;*/
  /*  jj += 1;*/
  /*}else if(w<13){*/
  /*  jj += 1;*/
  /*}else if(w<15){*/
  /*  jj += 1;*/
  /*  ii -= 1;*/
  /*}else{ // 16*/
  /*  ii -= 1;*/
  /*}*/

  atomicAdd(&hits[ii*grid_size+jj], 1);

  return;
}

__global__ void agg(
    const int n,
    const unsigned int grid_size,
    const bool *grid,
    const float *massx,
    const float *massy,
    const int *neigh,
    int *hits
    ){
  const unsigned int ij = blockIdx.x*THREADS + threadIdx.x;
  const unsigned int i = ij/grid_size;
  const unsigned int j = ij%grid_size;

  if (ij>=n){
    return;
  }
  if (neigh[ij]<1){
    return;
  }
  if (!grid[ij]){
    return;
  }

  _increment_hit_count(grid_size, i, j, grid, massx, massy, neigh, hits);

}
