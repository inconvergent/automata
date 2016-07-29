#define THREADS _THREADS_

__device__ void _increment_hit_count(
    const int grid_size,
    const int i,
    const int j,
    const bool *grid,
    const float *massx,
    const float *massy,
    const int *neigh,
    int *hits
    ){
  const float one = 1.0f/(float)grid_size;
  const int k = i*grid_size+j;
  const float x = (float)i*one+massx[k]*one;
  const float y = (float)j*one+massy[k]*one;

  const int ii = (int)round(x*grid_size);
  const int jj = (int)round(y*grid_size);

  atomicAdd(&hits[ii*grid_size+jj], 1);
  return;
}

__global__ void agg(
    const int n,
    const int grid_size,
    const bool *grid,
    const float *massx,
    const float *massy,
    const int *neigh,
    int *hits
    ){
  const int ij = blockIdx.x*THREADS + threadIdx.x;
  const int i = (int)floor(float(ij)/(float)grid_size);
  const int j = (ij-grid_size*i);

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
