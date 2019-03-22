__kernel void matrix_mult_three(__global const float *x,
                        __global const float *y,
                        __global float *restrict z,
                        const unsigned rows,
                        const unsigned cols)
{
  int index_x = get_global_id(0);
  int index_y = get_global_id(1);

  // Check edges
  if( (index_x + 3) >= cols || (index_y + 3) >= rows) {
    z[index_y*cols + index_x] = x[index_y*cols + index_x];
    return;
  }

  int acc = 0;
  z[index_y*cols + index_x] = 0.;
  for(int kr = 0; kr < 3; kr++) {
    for(int kc = 0; kc < 3; kc++) {
      z[index_y*cols + index_x] += (x[(index_y+kr)*cols + index_x + kc] * y[kr*3 + kc]);
    }
  }
}

__kernel void matrix_mult_three_sobel(__global const float *x,
                        __global const float *y,
                        __global float *restrict z,
                        const unsigned rows,
                        const unsigned cols)
{
  int index_x = get_global_id(0);
  int index_y = get_global_id(1);

  float soble_mat[] = { 3, 0, -3, 10, 0, -10, 3, 0, -3};
  float soble_mat_trans[] = { 3, 10, 3, 0, 0, 0, -3, -10, -3};

  // Check edges
  if( (index_x + 3) >= cols || (index_y + 3) >= rows) {
    z[index_y*cols + index_x] = x[index_y*cols + index_x];
    return;
  }

  int acc = 0;
  z[index_y*cols + index_x] = 0.;
  for(int kr = 0; kr < 3; kr++) {
    for(int kc = 0; kc < 3; kc++) {
      z[index_y*cols + index_x] += (x[(index_y+kr)*cols + index_x + kc] * soble_mat[kr*3 + kc]);
    }
  }

  for(int kr = 0; kr < 3; kr++) {
    for(int kc = 0; kc < 3; kc++) {
      z[index_y*cols + index_x] += (x[(index_y+kr)*cols + index_x + kc] * soble_mat_trans[kr*3 + kc]);
    }
  }
}
