__kernel void chmm(__global float* A, 
                   __global float* B, 
                   __global float* C, 
                   __local float* a_row_vec,
                   __local float* b_col_vec,
                   uint points_per_group, 
                   uint a_dim, 
                   uint b_col_dim, 
                   float alpha, 
                   float beta)
{
    uint local_id  = get_local_id(0);        // Thread local ID
    uint global_id = get_global_id(0);       // Thread global ID
    uint group_id  = get_group_id(0);        // Work group ID
    
    uint local_size = get_local_size(0);     // Local size or Work group size
    uint global_size = get_global_size(0);   // Global size or Number of Work items
    
    uint num_groups = get_num_groups(0);     // Number of Work groups

    if(global_id == 0)
    {
        printf("Global size = %d\n", global_size);
        printf("Local size = %d\n", local_size);
        printf("Number of groups = %d\n", num_groups);
    }
     
    //float temp = a_dim/local_size;
    //int iter_per_col = ceil(temp); 
    uint col_width = points_per_group/2;
    uint iter_per_col = (a_dim/col_width) + 1;
    uint col_done=0;
    
    float partial_sum=0;
    if(global_id < global_size)
    {
        //printf("column width : %d\n", col_width);
        //printf("iteration per column : %d\n", iter_per_col);
        
        for(int i=0; i<b_col_dim; ++i)
        {
            partial_sum = 0;
            col_done = 0;
            for(int itr=0; itr<iter_per_col; ++itr)
            {
                for(int j=0; j<col_width; ++j)
                {
                    a_row_vec[j] = A[j + itr*col_width + global_id*a_dim];
                    b_col_vec[j] = B[j*b_col_dim + itr*col_width + i]; 
                } 
                //for(int i=0; i<col_width; i++)
                //    printf("A[%d] = %f   ", i, A[i]);
                for(int k=0; k<col_width && col_done<a_dim; ++k)
                {
                    partial_sum += a_row_vec[k]*b_col_vec[k];
                    //printf("partial_sum = %f : %d\n", partial_sum, global_id);
                    ++col_done;
                }
                //printf("\n");
                //printf("\n");
            }
            
            //printf("partial_sum = %f : %d\n", partial_sum, global_id);
            C[i + global_id*b_col_dim] = partial_sum;
        }
    }
    
    
    
    /*float partial_sum=0;
    if(global_id < global_size)
    {
        printf("a_dim - b_col_dim : %d - %d\n", a_dim, b_col_dim);
        //for(int i=0; i<a_dim*b_col_dim; i++)
        //    printf("B[%d] = %f\n", i, B[i]);
        //for(int i=0; i<a_dim*a_dim; i++)
        //    printf("A[%d] = %f\n", i, A[i]);
        for(int i=0; i<b_col_dim; i++)
        {
            partial_sum = 0.0;
            for(int j=0; j<a_dim; j++)
            {
                //printf("A[%d] = %f", j, A[j + global_id*a_dim]);
                //printf("B[%d] = %f\n", j, B[j*b_col_dim * i]);
                partial_sum += A[j + global_id*a_dim]*B[j*b_col_dim + i];
                //printf("B[%d] = %f\n",((j*b_col_dim * i), B[j*b_col_dim * i]);
                //printf("%d\n", j);
            }
            //printf("partial_sum = %f : %d\n", partial_sum, global_id);
            C[i + global_id*b_col_dim] = partial_sum;            
        }
    }*/
    /*
    if(global_id == 0)
    {
        for(int i=0; i<a_dim*b_col_dim; i++)
        {
            printf("C[%d] = %f\n", i, C[i]);
        }
    }*/
    
    
}
