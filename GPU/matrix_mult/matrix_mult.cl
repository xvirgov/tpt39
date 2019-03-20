__kernel void vector_add(__global const float *x,
                        __global const float *y,
                        __global float *restrict z,
                        const unsigned N,
                        const unsigned M,
                        const unsigned P)
{
  // const unsigned row_N = *N;
  int index_x = get_global_id(0);
  int index_y = get_global_id(1);
  // int row_num = index_x %
  // printf("X::[%d],Y::[%d]\n", index_x, index_y);
  // // printf("BEF::%f = %f + %f\n", z[index]), x[index], y[index];
  // // z[index] = x[index] + y[index];
  // // printf("AFT::%f = %f + %f\n", z[index]), x[index], y[index];
  // float res = 0.;
  //
  // for()
  // printf("---------------\n");
  // printf("N::[%d],X_EL[%d,%d] = %f\n", N, index_y, index_x, x[index_y*N + index_x]);
  // printf("N::[%d],Y_EL[%d,%d] = %f\n", N, index_y, index_x, y[index_y*N + index_x]);
  z[index_y*N + index_x] = 0.;
  for(int j = 0; j < P; j++) {
    z[index_y*N + index_x] += (x[index_y*P+j] * y[j*M + index_x]);
    //  printf("INDEXES:[%d,%d] :: x[%d] --- [%f], y[%d] --- [%f] ---\n",  index_y, index_x, index_y*P+j, x[index_y*P+j], j*M + index_x, y[j*M + index_x]);
  }

  // printf("N::[%d],Z_EL[%d,%d] = %f\n", N, index_y, index_x, z[index_y*N + index_x]);
  // printf("---------------\n");
}
