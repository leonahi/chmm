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

//#define NUM_POINTS 1000

#define TEST_CODE

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
    unsigned int num_points, points_per_group;
    
    
    /* Data and Buffer */
    uint a_dim = A_DIM;
    uint b_row_dim = B_ROW_DIM;
    uint b_col_dim = B_COL_DIM;
    uint c_row_dim = C_ROW_DIM;
    uint c_col_dim = C_COL_DIM;
    uint alpha = ALPHA;
    uint beta = BETA;    
    float *A;
    float *B;
    float *X;
    float *C;
    
    cout << "Allocation space for A, B, C matrices.....";
    A = new float[a_dim*a_dim];
    B = new float[b_row_dim*b_col_dim];
    C = new float[c_row_dim*c_col_dim];
    X = new float[a_dim*b_col_dim];
    cout << "Done!!" << endl;
    
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
    
    /*cout << "Initializing A, B, C data matrices......";
    srand(time(NULL));
    for(uint i=0; i<(a_dim*a_dim); ++i) 
        A[i] = rand();
    for(uint i=0; i<(b_row_dim*b_col_dim); ++i)
        B[i] = rand();
    for(uint i=0; i<(c_row_dim*c_col_dim); ++i)
        C[i] = rand();
    cout << "Done!!" << endl;
    */
#ifdef TEST_CODE
    /* Checking for test code
     * 
     * A, B, C Initialized to 1
     * then X = aplha*A*B
     *      C = X + beta*C
     *      print C
     * 
     */
    for(int i=0; i<(a_dim*a_dim); i++) 
        A[i] = 1.0;
    for(int i=0; i<(b_row_dim*b_col_dim); i++)
    {
        B[i] = 1.0;
        C[i] = 1.0;
    }        
    //matrix_mult(A, B, X, a_dim, a_dim, b_col_dim, alpha);
    //matrix_add(X, C, c_row_dim, c_col_dim, beta); 
    //print_matrix(B, b_row_dim, b_col_dim);
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
                  (a_dim*a_dim)*sizeof(float), A, &err);
    //A_buffer = clCreateBuffer(context, 
    //              CL_MEM_READ_ONLY,
    //              (a_dim*a_dim)*sizeof(float), NULL, &err);
    if(err < 0)
    {
        perror("Couldn't create a buffer : A_buffer");
        exit(1);
    }
    
    B_buffer = clCreateBuffer(context, 
                  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                  (b_row_dim*b_col_dim)*sizeof(float), B, &err);
    //B_buffer = clCreateBuffer(context, 
    //              CL_MEM_READ_ONLY,
    //              (b_row_dim*b_col_dim)*sizeof(float), NULL, &err);
    if(err < 0)
    {
        perror("Couldn't create a buffer : B_buffer");
        exit(1);
    }

    C_buffer = clCreateBuffer(context, 
                  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                 (c_row_dim*c_col_dim)*sizeof(float), C, &err);
    //C_buffer = clCreateBuffer(context, 
    //              CL_MEM_READ_WRITE,
    //              (c_row_dim*c_col_dim)*sizeof(float), NULL, &err);
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
    local_size = (int)pow(2, trunc(log2(local_size)));
    cout << "Local size found: " << local_size << endl;
    cout << "Setting work group size (local_size) = " << local_size << endl;

    
    /* Determine preferred maximum work-group size */
    err = clGetKernelWorkGroupInfo(kernel, device, 
             CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
             sizeof(pref_workgoup_size), &pref_workgoup_size, NULL);
    if(err < 0) 
    {
        perror("Couldn't find the preferred maximum work-group size");
        exit(1);   
    };
    cout << "Preferred work group size found : " << pref_workgoup_size << endl;


   /* Determine local memory size */
   err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 
            sizeof(local_mem_size), &local_mem_size, NULL);
   if(err < 0) 
   {
       perror("Couldn't determine the local memory size");
       exit(1);   
   };
   cout << "Local memory size found : " << local_mem_size << endl;
   
   points_per_group = local_mem_size/sizeof(float);  // Load row of A and col of B in local memory 
   cout << "Setting points per group : " << points_per_group << endl;
   
   if(points_per_group > a_dim)
       points_per_group = a_dim;
 
   /* Set kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_buffer);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_buffer);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_buffer);
   err |= clSetKernelArg(kernel, 3, local_mem_size/2, NULL);
   err |= clSetKernelArg(kernel, 4, local_mem_size/2, NULL);
   err |= clSetKernelArg(kernel, 5, sizeof(points_per_group), &points_per_group);
   err |= clSetKernelArg(kernel, 6, sizeof(a_dim), &a_dim);
   err |= clSetKernelArg(kernel, 7, sizeof(b_col_dim), &b_col_dim);
   err |= clSetKernelArg(kernel, 8, sizeof(alpha), &alpha);
   err |= clSetKernelArg(kernel, 9, sizeof(beta), &beta);
   if(err < 0)
   {
       cout << "Unable to set kernel argument" << endl;
       exit(1);
   }
      
   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };
   
   
   /*err = clEnqueueWriteBuffer(queue, A_buffer, 
                                 CL_TRUE, 0, 
                                 sizeof(float)*(a_dim*a_dim), 
                                 A, 0, NULL, NULL);
   if(err < 0) 
   {
      perror("Unable to write buffer");
      exit(1);
   }
   
   err = clEnqueueWriteBuffer(queue, B_buffer, 
                                 CL_TRUE, 0, 
                                 sizeof(float)*(b_row_dim*b_col_dim), 
                                 B, 0, NULL, NULL);
   if(err < 0) 
   {
      perror("Unable to write buffer");
      exit(1);
   }*/

   
   err = clEnqueueWriteBuffer(queue, C_buffer, 
                                 CL_TRUE, 0, 
                                 sizeof(float)*(c_row_dim*c_col_dim), 
                                 C, 0, NULL, NULL);
   if(err < 0) 
   {
      perror("Unable to write buffer");
      exit(1);
   }
   
      
   
   //global_size = a_dim*b_col_dim;
   cl_event event;
   global_size = a_dim;
   local_size = 1;
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                &global_size, &local_size,
                                0, NULL, &event);
   if(err < 0)
   {
       perror("Couldn't enqueue the kernel");
       exit(1);
   }
   
   clFinish(queue);

//---------------------------------------------------------------------------------------  
// Profiling data
//---------------------------------------------------------------------------------------  
  cl_ulong ev_start_time = (cl_ulong)0;
  cl_ulong ev_end_time   = (cl_ulong)0;
  size_t ret_size;
  
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  if(err < 0)
  {
      perror("Error: clGetEventProfilingInfo");
      exit(1);
  }
  
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);
  if(err < 0)
  {
      perror("Error: clGetEventProfilingInfo");
      exit(1);
  }

   
   
   /* Read the results */
   err = clEnqueueReadBuffer(queue, C_buffer, CL_TRUE, 0, 
         (c_row_dim*c_col_dim)*sizeof(float), C, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
   }
   

   cl_ulong run_time = ev_end_time - ev_start_time;
  
   cout<<"Run time : "<< run_time << " ns" << endl;

   
   
   /* Deallocate resources */
   cout << "Releasing all the resources....";
   clReleaseMemObject(A_buffer);
   clReleaseMemObject(B_buffer);
   clReleaseMemObject(C_buffer);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   
   delete [] A;
   delete [] B;
   delete [] C;
   cout << "Done!!" << endl;
   
    return 0;
}