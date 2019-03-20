__kernel void vector_add(__global const float *x,
                        __global const float *y,
                        __global float *restrict z)
{
  int index = get_global_id(0);
  // printf("BEF::%f = %f + %f\n", z[index]), x[index], y[index];
  z[index] = x[index] + y[index];
  // printf("AFT::%f = %f + %f\n", z[index]), x[index], y[index];
}
