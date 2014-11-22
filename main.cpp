#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

#include <CL/cl.h>

#include "OPENCL_global_func.hpp"
#include "app_specific.hpp"

#define PROGRAM_FILE "chmm.cl"
#define KERNEL_FUNC "chmm"

#define NUM_POINTS 1000

//#define TEST_CODE

extern void matrix_mult(float* A, float* B, float* C, int a_row_dim, int a_col_dim, int b_col_dim, float alpha_const);
extern void matrix_add(float* X, float* C, int c_row_dim, int c_col_dim,float beta_const);
extern void print_matrix(float* C, int c_row_dim, int c_col_dim);


int main()
{
    /* Host/Device data structure */
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;
    size_t global_size, local_size, pref_workgoup_size;
    cl_ulong local_mem_size;
    
    /* Data and Buffer */
    unsigned int num_points, points_per_group;
    float A[(A_dim*A_dim)];
    float B[(B_row_dim*B_col_dim)];
    float X[A_dim*B_col_dim];
    float C[(C_row_dim*C_col_dim)];
    
    cl_mem A_buffer, B_buffer, C_buffer;
    
    /* Initialize data */
    /*srand(time(NULL));
    for(int i=0; i<(A_dim*A_dim); i++) 
    {
        A[2*i] = rand();
        A[2*i+1] = rand();
    }
    for(int i=0; i<(B_row_dim*B_col_dim); i++)
    {
        B[i] = rand();
    }*/
    
    srand(time(NULL));
    for(int i=0; i<(A_dim*A_dim); ++i) 
        A[i] = rand();
    for(int i=0; i<(B_row_dim*B_col_dim); ++i)
        B[i] = rand();
    for(int i=0; i<(C_row_dim*C_col_dim); ++i)
        C[i] = rand();
    
    
#ifdef TEST_CODE
    /* Checking for test code
     * 
     * A, B, C Initialized to 1
     * then X = aplha*A*B
     *      C = X + beta*C
     *      print C
     * 
     */
    for(int i=0; i<(A_dim*A_dim); i++) 
        A[i] = 1;
    for(int i=0; i<(B_row_dim*B_col_dim); i++)
    {
        B[i] = 1;
        C[i] = 1;
    }        
    matrix_mult(A, B, X, A_dim, A_dim, B_col_dim, alpha);
    matrix_add(X, C, C_row_dim, C_col_dim, beta); 
    print_matrix(C, C_row_dim, C_col_dim);
#endif    
    
    
    /* Create a device */
    device = create_device();
    
    /* Create context */
    context =  clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err < 0) {
        perror("Couldn't create a context");
        exit(1);   
    }    
    
    /* Build the program */
    program = build_program(context, device, PROGRAM_FILE);

    /* Create kernel for the chmm */
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if(err < 0) {
        printf("Couldn't create the KERNEL_FUNC kernel: %d", err);
        exit(1);
    };
    
    /* Create buffer */
    A_buffer = clCreateBuffer(context, 
                  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                  A_dim*A_dim*sizeof(float), A, &err);
    if(err < 0)
    {
        perror("Couldn't create a buffer : A_buffer");
        exit(1);
    }
    
    B_buffer = clCreateBuffer(context, 
                  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                  B_row_dim*B_col_dim*sizeof(float), A, &err);
    if(err < 0)
    {
        perror("Couldn't create a buffer : B_buffer");
        exit(1);
    }

    C_buffer = clCreateBuffer(context, 
                  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                  C_row_dim*C_col_dim*sizeof(float), A, &err);
    if(err < 0)
    {
        perror("Couldn't create a buffer : C_buffer");
        exit(1);
    }
    
    /* Determine maximum work-group size */
    err = clGetKernelWorkGroupInfo(kernel, device, 
             CL_KERNEL_WORK_GROUP_SIZE, 
             sizeof(local_size), &local_size, NULL);
    if(err < 0) 
    {
        perror("Couldn't find the maximum work-group size");
        exit(1);   
    };
    cout << "Local size found: " << local_size << endl;

    /* Determine preferred maximum work-group size */
    err = clGetKernelWorkGroupInfo(kernel, device, 
             CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
             sizeof(pref_workgoup_size), &pref_workgoup_size, NULL);
    if(err < 0) 
    {
        perror("Couldn't find the preferred maximum work-group size");
        exit(1);   
    };
    cout << "Preferred work group size : " << pref_workgoup_size << endl;

    cout << "Work group size : " << (int)pow(2, trunc(log2(local_size))) << endl;

   /* Determine local memory size */
   err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 
            sizeof(local_mem_size), &local_mem_size, NULL);
   if(err < 0) 
   {
       perror("Couldn't determine the local memory size");
       exit(1);   
   };
   cout << "Local memory size : " << local_mem_size << endl;
   
    
    return 0;
}