import numpy as np
import matplotlib.pyplot as plt



def scalar_vector(a, x):
    '''
   

    This function calculates the product beteween a scalar "a" and a vector "x"
    
    ...
    
    input
    a -> float - scalar
    x -> numpy array - vector
    
    ...
    
    output
    
    y -> numpy array - vector
    
    '''
    y=np.zeros_like(x)
    
    for i in range(x.size):
        y[i]=a*x[i]

    return y


def dot_product(x,y):
    '''
    Calculates the inner product between two vectors, 'x' and 'y'
    ...
    input
    x-> numpy array - vector
    y-> numpy array - vector
    output
    c-> float - scalar (dot product of x*y)
    '''
    assert x.size==y.size, 'The vectors need to have the same dimensions'
    c = 0.0
    for i in range(x.size):
        c = c + x[i]*y[i]
        
    return c


def hadamard(x,y):

    '''
    Calculates the hadamard product between two vectors, 'x' and 'y'
    ...

    input

    x-> numpy array - vector
    y-> numpy array - vector

    output
    r-> numpy array - vector (hadamard product of x*y)

    '''
    assert x.size==y.size, 'The vectors need to have the same dimensions'

    r=np.zeros_like(x)
    for i in range(x.size):
        r[i]= x[i]*y[i]
        
    return r



def outer_product_1(x,y):

    '''
    Calculates the outer product between two vectors, 'x' and 'y' using "for notation".
    ...

    input

    x-> numpy array - vector of dimension N
    y-> numpy array - vector of dimension M

    output
    z-> numpy array - Matrix of dimension NXM (x*y^T)

    '''
    z=np.zeros([x.size,y.size])
    for i in range(x.size):
        for j in range(y.size):
            z[i,j]=z[i,j] + x[i]*y[j]
            
    return z


def outer_product_2(x,y):
    '''
    Calculates the outer product between two vectors, 'x' and 'y' using "collon notation" by a row partition.
    ...

    input

    x-> numpy array - vector of dimension N
    y-> numpy array - vector of dimension M

    output
    z-> numpy array - Matrix of dimension NXM (x*y^T)

    '''
    
    z=np.zeros([x.size,y.size])
    for i in range(x.size):
        z[i,:]=z[i,:] + x[i]*y[:]
    return z


def outer_product_3(x,y):
    '''
    Calculates the outer product between two vectors, 'x' and 'y' using "collon notation" by a columns partition.
    ...

    input

    x-> numpy array - vector of dimension N
    y-> numpy array - vector of dimension M

    output
    z-> numpy array - Matrix of dimension NXM (x*y^T)

    '''
    
    z=np.zeros([x.size,y.size])
    for j in range(y.size):
        z[:,j]=z[:,j] + x[:]*y[j]
    return z



def matrix_vector1(A,x):
    '''
    Calculates the product between matrix and vector using doubly nested for.
    ...

    input

    A-> numpy array - Matrix of dimension NxM
    x-> numpy array - vector of dimension M

    output
    y-> numpy array - vector of dimension N 

    '''  
    assert A.shape[1]==x.size, 'The vectors must have the same dimensions of the columns of A'
    
    y=np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(x.size):
            y[i] += A[i,j]*x[j]
            
    return y


def matrix_vector2(A,x):
    '''
    Calculates the product between matrix and vector using dot product formulation.
    ...

    input

    A-> numpy array - Matrix of dimension NxM
    x-> numpy array - vector of dimension M

    output
    y-> numpy array - vector of dimension N 

    '''  
    assert A.shape[1]==x.size, 'The vectors must have the same dimensions of the columns of A'
    
    y=np.zeros(x.size)
    for i in range(A.shape[0]):
        y[i]=dot_product(A[i,:],x[:])
            
    return y



def matrix_vector3(A,x):
    '''
    Calculates the product between matrix and vector using linear combination formulation.
    ...

    input

    A-> numpy array - Matrix of dimension NxM
    x-> numpy array - vector of dimension M

    output
    y-> numpy array - vector of dimension N 

    '''  
    assert A.shape[1]==x.size, 'The vectors must have the same dimensions of the columns of A'
    
    y=np.zeros(x.size)
    for j in range(x.size):
        y[:]+=A[:,j]*x[j]
            
    return y

def sma_artur(x, window):
    '''
    it calculates the moving average 
    
    Input
    
    x -> numpy array - vector of dimension N
    window -> integer scalar - lenght of moving window 
    
    Output
        
    x_f -> numpy array - vector filtered data
    
    '''
    assert x.size >= window, 'The vector should be more gran than the window'
    assert window%2 != 0, 'The lenght of the window  should be odd'
    assert window >= 3, 'increase the window_size'

    dim = x.size
    i0 = window//2
    
    A = np.array(np.hstack(((1./window)*np.ones(window), np.zeros(dim - window + 1))))
    A = np.resize(A, (dim-2*i0, dim))
    A = np.vstack((np.zeros(dim), A, np.zeros(dim)))
    x_f = np.empty_like([x])
    x_f = matrix_vector1(A,x)
    return x_f

def sma1d_vanderlei(data, window_size):
    '''
    Apply a simple moving average filter with
    size window_size to data.
    
    input
    data: numpy array 1D - data set to be filtered.
    window_size: int - number of points forming the window.
                 It must be odd. If not, it will be increased
                 by one.
                 
    output
    filtered_data: numpy array 1D - filtered data. This array has the
                   same number of elementos of the original data.
    '''
    
    assert data.size >= window_size, \
        'data must have more elements than window_size'
    
    assert window_size%2 != 0, 'window_size must be odd'

    assert window_size >= 3, 'increase the window_size'

    # lost points at the extremities
    i0 = window_size//2

    # non-null data
    N = data.size - 2*i0

    filtered_data = np.empty_like(data)

    filtered_data[:i0] = 0.
    filtered_data[-1:-i0-1:-1] = 0.

    for i in range(N):
        filtered_data[i0+i] = np.mean(data[i:i+window_size])
        
    return filtered_data

def derivative(y,h):
    '''
    it calculates the first derivate using matrix-vector product 
    for representing the cetral finite difference equation  
    df(xi)/dx can be by calculates Dy product.
    
    Input
    
    y -> numpy array - vector of datas , dimension N
    h -> integer scalar - spacing of x 
    
    Output
        
    fd -> numpy array - vector of the first derivate
    
    '''
    
    dim = y.size
    
    
    fd = np.empty_like([y])
     
    D1 = np.array(np.hstack((([-1.,0.,1], np.zeros(dim-2)))))
    D2 = np.resize(D1,((dim-2,dim)))
    D3 = np.vstack((np.zeros(dim), D2, np.zeros(dim)))
    D4 = (1./(2*h))*D3
    fd = matrix_vector1(D4,y)
    return fd




def mat_mat1(A,B):
    '''
    it receives the A and B matrices (A nxm, Bmxl) and then it calculates a third matrix as result of the product between A and B
    using triply nested for.
    '''
    '''    
        input

    A-> numpy array - Matrix of dimension NxM
    B-> numpy array - Matrix of dimension MxL

    output
    C-> numpy array - Matrix of dimension NxL 
 
    '''
    assert A.shape[1] == B.shape[0], 'The number of columns of A is different of the number of B rows'   
    
    C = np.zeros([A.shape[0],B.shape[1]])

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = 0.0
            for k in range(A.shape[1]):
                C[i,j] +=A[i,k]*B[k,j]
    return C
    

def mat_mat2(A,B):
    '''
    it receives the A and B matrices (A nxm, Bmxl) and then it calculates a third matrix as result of the product between A and B
    using dot production formulation.
    
        input

    A-> numpy array - Matrix of dimension NxM
    B-> numpy array - Matrix of dimension MxL

    output
    C-> numpy array - Matrix of dimension NxL 

    '''  
    
    assert (A.shape[1]) == (B.shape[0]), 'The number of columns of A is different of the number of B rows'   
    
    C = np.zeros([A.shape[0],B.shape[1]])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = 0.0
            C[i,j] = dot_product(A[i,:],B[:,j])
    
    return C        

def mat_mat3(A,B):
    '''
    it receives the A and B matrices (A nxm, Bmxl) and then it calculates a third matrix as result of the product between A and B
    using using dot formulation with row partition.
    
        input

    A-> numpy array - Matrix of dimension NxM
    B-> numpy array - Matrix of dimension MxL

    output
    C-> numpy array - Matrix of dimension NxL 

    '''  
    
    assert A.shape[1] == B.shape[0], 'The number of columns of A is different of B'
    C=np.zeros([A.shape[0],B.shape[1]])
    for i in range(A.shape[0]):
        C[i,:]= matrix_vector1(B[:,:].T,A[i,:])
           
    return C              







def mat_mat4(A,B):
    '''
        it receives the A and B matrices (A nxm, Bmxl) and then it calculates a third matrix as result of the product between A and B
    using using dot formulation with row partition.
    
        input

    A-> numpy array - Matrix of dimension NxM
    B-> numpy array - Matrix of dimension MxL

    output
    C-> numpy array - Matrix of dimension NxL 
    '''
    assert A.shape[1]==B.shape[0], 'The columnss number is different of columnss A'
                  
    C=np.zeros([A.shape[0],B.shape[1]]) 
    
    for j in range(B.shape[1]):
        C[:,j]=matrix_vector1(A[:,:],B[:,j])

    return C             

def mat_mat5(A,B):
    '''
    it receives the A and B matrices (A nxm, Bmxl) and then it calculates a third matrix as result of the product between A and B
    using using outer product formulation.
    
        input

    A-> numpy array - Matrix of dimension NxM
    B-> numpy array - Matrix of dimension MxL

    output
    C-> numpy array - Matrix of dimension NxL 

    '''  
    
    assert A.shape[1] == B.shape[0], 'The number of columns of A is diber of'
    C=np.zeros([A.shape[0],B.shape[1]])
    for k in range(A.shape[1]):
        C[:,:]+= outer_product_1(A[:,k],B[k,:])
           
    return C  



def rotation_matrix1(theta):
    '''
    it receives an angle value and return a matrix termed: matrix of rotation
    
    
    input
    
    theta -> escalar
    
    
    output
    
    R4 -> numpy array - Matrix of rotation 3x3
    
    '''
      
        
    R1 = np.array([1.,0.,0.])
    R2 = np.array([0., np.cos(theta), np.sin(theta)])
    R3 = np.array([0.,(-1)*np.sin(theta), np.cos(theta)])
    
    R4 = np.vstack((R1,R2))
    
    R5 = np.vstack((R4,R3))
    
    
    return R5
    
def rotation_matrix2(theta):
    '''
    it receives an angle value and return a matrix termed: matrix of rotation
    
    
    input
    
    theta -> escalar
    
    
    output
    
    R4 -> numpy array - Matrix of rotation 3x3
    
    '''
      
        
    R1 = np.array([np.cos(theta),0.,(-1)*np.sin(theta)])
    R2 = np.array([0., 1., 0.])
    R3 = np.array([np.sin(theta),0. , np.cos(theta)])
    
    R4 = np.vstack((R1,R2))
    
    R5 = np.vstack((R4,R3))
    
    
    return R5    


def rotation_matrix3(theta):
    '''
    it receives an angle value and return a matrix termed: matrix of rotation
    
    
    input
    
    theta -> escalar
    
    
    output
    
    R4 -> numpy array - Matrix of rotation 3x3
    
    '''
      
        
    R1 = np.array([np.cos(theta),np.sin(theta),0.])
    R2 = np.array([(-1)*np.sin(theta), np.cos(theta), 0])
    R3 = np.array([0.,0., 1.])
    
    R4 = np.vstack((R1,R2))
    
    R5 = np.vstack((R4,R3))
    
    
    return R5



def diag1(d,B):
    
    '''
    Calculates the product diagonal and full matrix
    
    input 
    
    d -> numpy array - vector with elements of the main diagonal of D
    B -> numpy array - full matrix (NxN)
    
    output
    
    C -> numpy array - matrix that is produced by d*B
    
    '''
    
    assert d.size==B.shape[0], 'dimensions wrong'
    C=np.zeros_like(B)
    
    
    for i in range(d.size):
        C[i,:]= d[i]*B[i,:]
        
    return C    




def diag2(d,B):
    
    '''
    Calculates the product diagonal and full matrix
    
    input 
    
    d -> numpy array - vector with elements of the main diagonal of D
    B -> numpy array - full matrix (NxN)
    
    output
    
    C -> numpy array - matrix that is produced by d*B
    
    '''

    assert d.size==B.shape[0], 'dimensions wrong'
    C=np.zeros_like(B)
    
    
    for j in range(d.size):
        C[:,j]= d[j]*B[:,j]
        
    return C 
    
def triang_3(U,x):


    assert x.size==U.shape[0], 'dimension is wrong'
    
    y=np.zeros_like(x)
                      
    for i in range(x.size):
        
        y[i]=dot_product(U[i,i:],x[i:])
        
    return y

def triang_5(U,x):


    assert x.size==U.shape[0], 'dimension is wrong'
    
    y=np.zeros_like(x)
                      
    for j in range(x.size):
        
        y[:j+1]+= U[:j+1,j]*x[j]
        
    return y


def triang_8(L,x):


    assert x.size==L.shape[0], 'dimension is wrong'
    
    z=np.zeros_like(x)
                      
    for i in range(x.size):
        
        z[i]=dot_product(L[i,:i+1],x[:i+1])
        
    return z

def triang_10(L,x):


    assert x.size==L.shape[0], 'dimension is wrong'
    
    z=np.zeros_like(x)
                      
    for j in range(x.size):
        
        z[j:] += L[j:,j]*x[j]
        
    return z


def U2ur(U):
    'Transforms the upper triangular matrix U into \
        a vector ur by using a row scheme'
    assert U.shape[0] == U.shape[1], 'U must be square'
    # indices of the non-null elements
    i, j = np.triu_indices(U.shape[0])
    # create the vector uc
  
    ur = U[i, j]
    
    return ur


def storage_U1(ur,x):
    
    
    y = np.zeros_like(x)
    N=x.size
    for i in range(x.size):
  
        k1 = i + i*(2*N - i - 1)/2
        
        k2 = N - 1 + i*(2*N - i - 1)/2
        
        y[i] = np.dot(ur[k1:k2+1],x[i:])
    
    return y

def U2uc(U):
    'Transforms the upper triangular matrix U into \
a vector uc by using a column scheme'
    assert U.shape[0] == U.shape[1], 'U must be square'
    # indices of the non-null elements
    i, j = np.triu_indices(U.shape[0])
    # reorganize the elements according to the column scheme
    p = np.argsort(j)
    i = i[p]
    j = j[p]
    # create the vector uc
    uc = U[i, j]

    return uc

def storage_U2(uc,x):
    '''
    Calculates the product of an upper triangular matrix U and a vector x by using a column-based storage scheme for U (CBU)
    (algorithm 5)

    input

    uc: numpy array - vector
    x: numpy array - vector

    output

    y: numpy array - vector
    '''

    y = np.zeros_like(x)
    

    for j in range(x.size):
        k1 = (j + 1)*j/2
        k2 = j + (j + 1)*j/2
        y[:j+1] = y[:j+1] + uc[k1:k2+1]*x[j]

    return y



def L2lc(L):
    'Transforms the lower triangular matrix L into \
a vector lc by using a column scheme'
    assert L.shape[0] == L.shape[1], 'L must be square'
    # indices of the non-null elements
    i, j = np.tril_indices(L.shape[0])
    # reorganize the elements according to the column scheme
    p = np.argsort(j)
    i = i[p]
    j = j[p]
    # create the vector lc
    lc = L[i, j]
    return lc
    

def storage_L1(lc,x):
    
    
    z = np.zeros_like(x)
    N = x.size

    for j in range(x.size):
        k1 = j + j*(2*N - j - 1)/2
        k2 = N - 1 + j*(2*N - j - 1)/2
        z[j:] = z[j:] + lc[k1:k2+1]*x[j]

    return z


def L2lr(L):
    'Transforms the lower triangular matrix L into \
a vector lc by using a row scheme'
    assert L.shape[0] == L.shape[1], 'L must be square'
    # indices of the non-null elements
    i, j = np.tril_indices(L.shape[0])
    # create the vector lr
    lr = L[i, j]
    return lr
  
    
def storage_L2(lr,x):
    
    

    z = np.zeros_like(x)


    for i in range(x.size):
        k1 = (i + 1)*i/2
        k2 = i + (i + 1)*i/2
        z[i] = dot_product(lr[k1:k2+1],x[:i+1])

    return z   
    
    
    
def symmetric(s,x):

    '''
    Calculates the product of a symmetric matrix and a vector x by usin a CBL or RBU scheme for S

    input

    s -> numpy array - vector
    x -> numpy array - vector

    output

    y -> numpy array - vector
    '''

    y = np.zeros(x.size)
    N = x.size

    for i in range(N):
        k1 = i + i*(2*N - i - 1)/2
        k2 = N - 1 + i*(2*N - i - 1)/2
        # Y[i] for main diagonal
        y[i] = y[i] + s[k1]*x[i]
        # y[i] for upper triangle
        y[i] = y[i] + dot_product(s[k1+1:k2+1],x[i+1:])
        # y[i] for lower triangle
        y[i+1:] = y[i+1:] + s[k1+1:k2+1]*x[i]

    return y    
    


def system_triup(U,y):

    
    '''
    it solves a linear system Ax=y, where A is a upper (U) matrix and y is a known vector

    input

    U -> numpy array - Matrix
    y -> numpy array - vector

    output

    x -> numpy array - vector
    '''
    assert U.shape[0] == U.shape[1], ' The matrix must be square'
    assert U.shape[0] == y.size ,'The two arrays need to have same dimensions'
    
    N = y.size
    x = np.zeros_like(y)
    
    for i in range(N-1,-1,-1):
        x[i] = y[i] - dot_product(U[i,i+1:] , x[i+1:])
        x[i] = x[i]/U[i,i]
        
    return x    

def system_trilow(L,y):
    

    '''
    it solves a linear system Ax=y, where A is a lower matrix (L) and y is a known vector

    input

    L -> numpy array - Matrix
    y -> numpy array - vector

    output

    x -> numpy array - vector
    '''
    assert L.shape[0] == L.shape[1], ' The matrix must be square'
    assert L.shape[0] == y.size ,'The two arrays need to have same dimensions'
    
    N = y.size
    x = np.zeros_like(y)
    
    for i in range(N):
        x[i] = y[i] - dot_product(L[i,:i] , x[:i])
        x[i] = x[i]/L[i,i]
        
    return x

def vsprofile(dz0,dz,t):
    
    
    '''
    it solves a linear system Ms=t, where M is a slowness matrix  t is a time of arrive 
    and dz e dz0 are distance between top and bottom of a layer

    input

    dz0 -> scalar - distance between the surface and the bottom of the first layer 
    dz -> scalra - constant distance of he subsequents layers 
    t -> numpy array - vector contain the time of travel 
    output

    s -> numpy array - vector contain the time of travel 
    '''
    
    
    s=np.zeros_like(t)
    
    N=t.size
    
    s[0]=t[0]/dz0 
    
    for i in range(1,N):
        
        s[i]= (t[i]-t[i-1])/dz
        
    return s

def permut (C, i):
    
    '''
    It produces one permutation in the C matrix avalauting the gran number in range of one row 
    
    input 
    
    C - numpy array - matrix
    i -scalar - 
    
    output
    
    p - numpy array - vector contains the order of permuatation
    C [p,:] - numpy array - permuted matrix
    '''
    p = [j for j in range(C.shape[0])]
    imax = i + np.argmax(np.abs(C[i:,i]))
    if imax != i:
        p[i], p[imax] = p[imax], p[i]
    return p, C[p,:]


def gauss_pivotation(A,y):
    
    '''The Gaussian elimination iteratively transforms an unstructured linear system
    Ax=y into an equivalent triangular system Bx=z having the same solution x.
    the pivots needed for computing the Gauss multipliers must be nonzero. 
    
    input
    
    A -> Numpy array - matrix
    
    y -> Numpy array - vector
    
    output
    
    x -> numpy array - vector


    '''

    assert A.shape[1]==y.size, 'The colmns of A must be equal to y size'

    N=y.size
    #C=np.zeros(N,N+1)
    
     
    C = np.vstack([A.T, y]).T
                 
    for k in range(N-1):
        # permutation step (computation of C tilde)
        p, C = permut(C, k)

        # assert the pivot is nonzero
        assert C[k,k] != 0., 'null pivot!'

        # calculate the Gauss multipliers and store them 
        # in the lower part of C
        C[k+1:,k] = C[k+1:,k]/C[k,k]

        # zeroing of the elements in the ith column
        C[k+1:,k+1:] = C[k+1:,k+1:] - outer_product_1(C[k+1:,k], C[k,k+1:])

    return C[:,:N], C[:,N]



    def decomposicao_LU(A):
        '''
        This algorithm receives a square matrix A and returns a matrix C containing the triangular matrices L and U.
        The elements of L, except the unitary elements of its main diagonal, are stored below the main diagonal of C.
        The elements of U are stored in the upper part of C, including its main diagonal.
        '''
        N = A.shape[0]
        C = np.copy(A)

        for i in range(1,N-1):

    # assert the pivot is nonzero
            assert C[i,i] != 0., 'null pivot!'

    # calculate the Gauss multipliers and store them 
    # in the lower part of C
            C[i+1:,i] = C[i+1:,i]/C[i,i]

    # zeroing of the elements in the ith column
            C[i+1:,i+1:] = C[i+1:,i+1:] - outer_product_1(C[i+1:,i], C[i,i+1:])

    return C



    def solve_system_lu(A,y):
    
       '''it solveS system Ax=y using LU decomposition and solution for a upper and lower triangular system 
    input

    A -> numpy array - Matrix
    y -> numpy array - vector

    output

    x -> numpy array - vector (solution vector)

    '''
    LU=np.copy(A)

    w=np.zeros_like(y)

    LU=deccompsicao_LU(A)

    U=np.triu(LU) 

    L=np.tril(LU)

    w=system_trilow(L,y)

    x=system_triup(U,w)        

    return x


    def Lu_pivoting(A,y):
        '''
        This algorithm receives a square matrix AA and returns the permutation matrix PP and a matrix CC containing the triangular matrices LL and UU. 
        The elements of LL, except the unitary elements of its main diagonal, 
        are stored below the main diagonal of CC. The elements of UU are stored in the upper part of CC, including its main diagonal.
        '''

        N = y.size
        C = np.copy(A)
        P = np.identity(N)

        for i in range(1,N-1):

        # permutation step (computation of C tilde)
            p, C = permut(C, i)
            P = P[p]

            # assert the pivot is nonzero
            assert C[i,i] != 0., 'null pivot!'

        # calculate the Gauss multipliers and store them 
        # in the lower part of C
            C[i+1:,i] = C[i+1:,i]/C[i,i]

    # zeroing of the elements in the ith column
            C[i+1:,i+1:] = C[i+1:,i+1:] - outer_product_1(C[i+1:,i], C[i,i+1:])

        return P, C




    def solves_sysLU_pivoting(A,y):

        '''

        input

        A -> numpy array - Matrix
        y -> numpy array - vector

        output

        x -> numpy array - vector (solution vector)

        '''
        C=np.copy(A)

        w=np.zeros_like(y)

        p,C=Lu_pivoting(A,y)

        U=np.triu(C) 

        L=np.tril(C)

        py=matrix-vector(p,y)


        w=system_trilow(L,py)

        x=system_triup(U,w)        

    return x





    def decom_cholesky(A):
 
        N=A.shape[0]
        G=np.copy(A)


        for j in range(N):

            G[j,j] = A[j,j] - dot_product(G[j,:j-1],G[j,:j-1])    

            G[j,j] =np.sqrt(G[j,j])

            G[j+1:,j] = (A[j+1:,j] - dot_product(G[j+1:,:j-1], G[j,:j-1]))/G[j,j]

    return G



    def solves_cholesky(A,y):
        

        '''
        '''

        G=np.copy(A)

        G=decom_cholesky(A)

        Gt=G.T

        W=np.zeros_like(y)

        x=np.zeros_like(y)

        w=fa.system_trilow(G,y)

        x=fa.system_triup(Gt,w)


    return x    


#T(z)= A(1- erf(z_i/sqrt(4*lamba*t)))

        
############################################################################################
   
    def integrand(t):
        result = (2./np.sqrt(np.pi))*np.exp(-t*t)
    return result


    def comp_trapezoidal(N, x0, x1, integrand):
        assert x1 > x0, 'x1 must be greater than x0'
        h = (x1 - x0)/(N - 1)
        t = np.linspace(x0, x1, N)
        f = integrand(t)
        result = h*(f[0]*0.5 + np.sum(f[1:-2]) + f[-1]*0.5)
    return result


    def geothermal_perturbation(A,years,z0,zn,N):
        
        '''cauculates the temperature in depth give a time
        after a perturbation of amplitude A'''
        
        x0=z0/np.sqrt(31.5576*years)
        x1=zn/np.sqrt(31.5576*years)
        
        Tz=A*(1-comp_trapezoidal(N,x0,x1,integrand))
            
            
    return Tz            


    def DFT_slow_(gk):
        """Compute the discrete Fourier Transform of the 1D array x"""
        
        gk = np.asarray(gk, dtype=float)
        Gn = np.zeros_like(gk)
        N = gk.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j*np.pi*k*n/N)
        Gn=np.dot(M,gk)
        
    return Gn


    def DFTI_slow_(Gn):
        """Compute the discrete Fourier Transform of the 1D array x"""
        
        Gn = np.asarray(Gn, dtype=float)
        gk = np.zeros_like(Gn)
        N = Gn.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(2j*np.pi*k*n/N)
        gk=(np.dot(M,Gn)/N)
        
    return Gn


            
            
                
        
