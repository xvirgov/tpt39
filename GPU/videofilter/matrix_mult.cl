__kernel void matrix_mult_three(__global const float *x,
                        __global const float *y,
                        __global float *restrict z)
{
  // Without tiling
  // int index_x = get_global_id(0);
  // int index_y = get_global_id(1);
  //
  // z[index_y*N + index_x] = 0.;
  // for(int j = 0; j < P; j++) {
  //   z[index_y*N + index_x] += (x[index_y*P+j] * y[j*M + index_x]);
  // }

  // With tiling
  // int index_x = get_group_id(0)*get_local_size(0) + get_local_id(0);
  // int index_y = get_group_id(1)*get_local_size(1) + get_local_id(1);
  //
  // z[index_y*N + index_x] = 0.;
  // for(int j = 0; j < P; j++) {
  //   z[index_y*N + index_x] += (x[index_y*P+j] * y[j*M + index_x]);
  // }
  z[0] = 0;
  printf("HELLO FROM GPU %f %f %f %d %d %d\n", x[0], y[0],z[0]);

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
