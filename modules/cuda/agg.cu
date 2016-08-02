#define THREADS _THREADS_

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
  const float one = 1.0f/(float)grid_size;
  const int k = i*grid_size+j;

  float x = (float)i*one+massx[k]*one;
  float y = (float)j*one+massy[k]*one;
  unsigned int ii = (unsigned int)round(x*grid_size);
  unsigned int jj = (unsigned int)round(y*grid_size);
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
