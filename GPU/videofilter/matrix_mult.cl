__kernel void matrix_mult_three(__global const float *x,
                        __global const float *y,
                        __global float *restrict z,
                        const unsigned rows,
                        const unsigned cols)
{
  // printf("%f ", y[0]);
  // Without tiling
  // int index_x = get_global_id(0);
  // int index_y = get_global_id(1);
  //
  // // border check
  // if(index_x + 3 >= cols) return;
  // if(index_y + 3 >= rows) return;
  // // 3-matrix check
  // if(index_x%3 || index_y%3) return;

  // printf("MAT start at [%d,%d]\n", index_x, index_y);

  // z[index_y*cols + index_x] = 0.;
  // printf("ROWS::%d,COLS::%d\n", rows, cols);
  // if(index_x + 3 >= rows) return;
  // if(index_y + 3 >= cols) return;
  // for(int j = 0; j < 3; j++) {
  //   for(int i = 0; i < 3; i++) {
  //     z[(index_y+j)*cols + index_x + i] += (x[(index_y+j)*cols+i] * y[j*3 + i]);
  //   }
  // }

  // for(int i = 0; i < 3; i++) {
  //   z[(index_y)*cols + index_x] += (x[(index_y+j)*cols+i] * y[j*3 + i]);
  // }
  // for(int yy = 0; yy < 3; yy++) {
  //   for(int xx = 0; xx < 3; xx++) {
  //     z[(index_y + yy)*cols + index_x + xx] = 0.;
  //     // printf("%d ", (index_y + yy)*cols + index_x + xx);
  //   }
  //   // printf("\n");
  // }

      // z[(index_y)*cols + index_x] = 100.;


int index_x = get_global_id(0);
int index_y = get_global_id(1);

// border check
if(index_x + 3 >= cols) return;
if(index_y + 3 >= rows) return;
// 3-matrix check
// if(index_x%3 || index_y%3) return;
// float *filter_mat_f=(float *) malloc(sizeof(float)*9);
// filter_mat_f[0] = 0.077847;
// filter_mat_f[1] = 0.123317;
// filter_mat_f[2] = 0.077847;
// filter_mat_f[3] = 0.123317;
// filter_mat_f[4] = 0.195346;
// filter_mat_f[5] = 0.123317;
// filter_mat_f[6] = 0.077847;
// filter_mat_f[7] = 0.123317;
// filter_mat_f[8] = 0.077847;
float filter_mat_f[] = { 0.077847,0.123317,0.077847,0.123317,0.195346,0.123317,0.077847,0.123317,0.077847 };
int acc = 0;
z[index_y*cols + index_x] = 0.;
// printf("%f ", y[0]);
for(int kr = 0; kr < 3; kr++) {
  for(int kc = 0; kc < 3; kc++) {
     z[index_y*cols + index_x] += (x[(index_y+kr)*cols + index_x + kc] * filter_mat_f[kr*3 + kc]);
    //  printf("%f ", y[kr*3 + kc]);
  }
}

//// matrix mult
// int index_x = get_global_id(0);
// int index_y = get_global_id(1);
//
// // border check
// if(index_x + 3 >= cols) return;
// if(index_y + 3 >= rows) return;
// // 3-matrix check
// if(index_x%3 || index_y%3) return;
    // for(int n = 0; n < 3; n++) {
    //   for(int m = 0; m < 3; m++) {
    //     z[(index_y + n)*cols + index_x + m] = 0;
    //     for(int jj = 0; jj < 3; jj++) {
    //       // z[(index_y + n)*cols + index_x + m] += (x[(index_y + n)*cols + index_x + m + jj] * y[jj*3+m]);
    //       z[(index_y + n)*cols + index_x + m] += x[(index_y + n)*cols + index_x + m + jj] *y[jj*3+m];
    //       printf("%f ", y[jj*3+m]);
    //       // z[(index_y + n)*cols + index_x + m] = x[(index_y + n)*cols + index_x + m + jj]*0.5;
    //     }
    //   }
    // }
    //// matrix mult


  // for(unsigned n = 0; n < N; n++) {
  //   for(unsigned m = 0; m < M; m++) {
  //     ref_output[n*N + m] = 0;
  //     for(unsigned jj = 0; jj < P; jj++) {
  //       ref_output[n*N + m] += (input_a[n*P+jj] * input_b[jj*M + m]);
  //     }
  //   }
  // }
// z[index_y*cols + index_x] = x[index_y*cols + index_x] * 0.5;
  // With tiling
  // int index_x = get_group_id(0)*get_local_size(0) + get_local_id(0);
  // int index_y = get_group_id(1)*get_local_size(1) + get_local_id(1);
  //
  // z[index_y*N + index_x] = 0.;
  // for(int j = 0; j < P; j++) {
  //   z[index_y*N + index_x] += (x[index_y*P+j] * y[j*M + index_x]);
  // }
 //  z[0] = 0;
 //  int index_x = get_global_id(0);
 //  int index_y = get_global_id(1);
 //
 // z[index_y*cols + index_x] = x[index_y*cols + index_x];
 // printf("%f ", z[index_y*cols + index_x]);

  //printf("GPU %f %f %f %d %d %d\n", x[0], y[0],z[0]);
  // printf("COORD::[%d,%d]\n", index_x, index_y);

  // For debugging
  // printf("index_x::[%d], index_y::[%d] ---> [%f]\n", index_x, index_y, z[index_y*N + index_x]);

  // printf("get_work_dim()::%d\n",get_work_dim());
  // printf("get_global_size(0)::%d\n",get_global_size(0));
  // printf("get_global_size(1)::%d\n",get_global_size(1));
  //
  // printf("get_global_id(0)::%d\n",get_global_id(0));
  // printf("get_global_id(1)::%d\n",get_global_id(1));
  //
  // printf("get_local_size(0)::%d\n",get_local_size(0));
  // printf("get_local_size(1)::%d\n",get_local_size(1));
  //
  // printf("get_local_id(0)::%d\n",get_local_id(0));
  // printf("get_local_id(1)::%d\n",get_local_id(1));
  //
  // printf("get_num_groups(0)::%d\n",get_num_groups(0));
  // printf("get_num_groups(1)::%d\n",get_num_groups(1));
  //
  // printf("get_group_id(0)::%d\n",get_group_id(0));
  // printf("get_group_id(1)::%d\n",get_group_id(1));
  //
  // printf("get_global_offset(0)::%d\n",get_global_offset(0));
  // printf("get_global_offset(1)::%d\n",get_global_offset(1));
}
