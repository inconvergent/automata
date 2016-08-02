#define THREADS _THREADS_

__device__ void _calculate_centre_of_mass(
    const unsigned int grid_size,
    const unsigned int i,
    const unsigned int j,
    const bool *grid,
    const unsigned int influence_rad,
    float *massx,
    float *massy,
    int *neigh
    ){
  const float one = 1.0f/(float)grid_size;

  int count = 0;

  float mx = 0.0f;
  float my = 0.0f;
  float nrm = 0.0f;

  const float x = i*one;
  const float y = j*one;
  float dx = 0;
  float dy = 0;

  const float rad = pow(one*(float)influence_rad, 2.0f);

  int k;
  for (unsigned int a=max(i-influence_rad,0);a<min(i+influence_rad+1,grid_size);a++){
    for (unsigned int b=max(j-influence_rad,0);b<min(j+influence_rad+1,grid_size);b++){
      k = a*grid_size+b;
      if (grid[k]){
        dx = x-a*one;
        dy = y-b*one;
        nrm = dx*dx+dy*dy;
        if (nrm>rad){
          continue;
        }
        mx += a*one;
        my += b*one;
        count += 1;
      }
    }
  }

  k = i*grid_size+j;
  neigh[k] = count;

  if (count>0){
    mx /= (float)count;
    my /= (float)count;

    mx = x-mx;
    my = y-my;

    nrm = mx*mx+my*my;
    if (nrm>0.0f){
      nrm = sqrt(nrm);
      massx[k] = mx/nrm;
      massy[k] = my/nrm;
    }
    else{
      massx[k] = 0.0f;
      massy[k] = 0.0f;
    }
  }
  else{
    massx[k] = 0.0f;
    massy[k] = 0.0f;
  }

  return;
}

__device__ void _count_connected(
    const unsigned int grid_size,
    const int i,
    const int j,
    const bool *grid,
    int *connected
    ){

  int k = i*grid_size+j;

  int count = 0;

  for (int a=max(i-1,0);a<min(i+2,grid_size);a++){
    for (int b=max(j-1,0);b<min(j+2,grid_size);b++){
      k = a*grid_size+b;
      if (grid[k]){
        count += 1;
      }
    }
  }
  k = i*grid_size+j;
  connected[k] = count;

  return;
}

__global__ void mass(
    const int n,
    const unsigned int grid_size,
    const bool *grid,
    const int influence_rad,
    float *massx,
    float *massy,
    int *neigh,
    int *connected
    ){
  const unsigned int ij = blockIdx.x*THREADS + threadIdx.x;
  const unsigned int i = ij/grid_size;
  const unsigned int j = ij%grid_size;

  if (ij>=n){
    return;
  }

  _calculate_centre_of_mass(grid_size, i, j, grid, influence_rad, massx, massy, neigh);
  _count_connected(grid_size, i, j, grid, connected);

}
