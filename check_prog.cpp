/* Check programm to test the functionality of parallel code.
 * 
 * Author: Nahit Pawar
 *
 */

#include <iostream>

using namespace std;

// C = alpha*A*B
void matrix_mult(float* A, float* B, float* C, int a_row_dim, int a_col_dim, int b_col_dim, float alpha_const)
{
    int dot_prod=0;
    for(int i=0; i<a_row_dim; ++i)
    {
        for(int j=0; j<b_col_dim; ++j)
        {
            dot_prod = 0;
            for(int k=0; k<a_row_dim; ++k)
                dot_prod += A[i*a_col_dim + k]*B[k*b_col_dim + j];
            C[i*b_col_dim + j] = alpha_const*dot_prod;
        }
    }
}

// C = X + beta*C
void matrix_add(float* X, float* C, int c_row_dim, int c_col_dim,float beta_const)
{
    for(int i=0; i<c_row_dim; ++i)
        for(int j=0; j<c_col_dim; ++j)
            C[i*c_col_dim + j] = X[i*c_col_dim + j] + beta_const*C[i*c_col_dim + j];
}

// Print matrix C
void print_matrix(float* C, int c_row_dim, int c_col_dim)
{
    for(int i=0; i<c_row_dim; ++i)
    {
        for(int j=0; j<c_col_dim; ++j)
            cout << C[i*c_row_dim + j] << " ";
        cout << endl;
    }
}
