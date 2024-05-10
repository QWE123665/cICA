from SPM import subspace_power_method
import numpy as np
from itertools import permutations
from scipy.optimize import minimize
from scipy.stats import uniform_direction
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import helper_functions
import pandas as pd
from scipy.optimize import minimize
from PyMoments import kstat
from itertools import product


# recovering mixing matrix using accurate cumulant tensors
def recovermixingmatrix_accurate(I,J,A,normal_variance):
    # generate the second and fourth cumulant tensors
    second_order_kstats = A[:,:-1]@ np.transpose(A[:,:-1])+normal_variance*A[:,-1].reshape(-1,1)@A[:,-1].reshape(1,-1)
    # fourth cumulant tensor
    fourth_order_kstats=helper_functions.generate_lowrank_tensor(A[:,:-1],4)
    # recover first J-1 columns via tensor decomposition
    cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
    def returnmindistancebewteenvectors(cols):
        def distancebewteenvectors(v1,v2):
            v1=v1.reshape(-1,1)
            v2=v2.reshape(-1,1)
            M=v1@np.transpose(v1)-v2@np.transpose(v2)
            return np.sum(M*M)
        lens=cols.shape[1]
        error=1
        for i in range(lens):
            for j in range(i+1,lens):
                error=min(error,distancebewteenvectors(cols[:,i],cols[:,j]))
        return error
    step=0
    while returnmindistancebewteenvectors(cols)<0.1 and step<100:
        cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
        step+=1
    
    # now we use second cumulant to find the last column of the mixing matrix 
    # by minimizing the distance bewteen a matrix in the linear span of the second cumulant matrix 
    # and all the rank 1 matrices M_1,...M_{J-1} obtained from the first J-1 columns of A
    rank1_matrixlist=[]
    for i in range(J-1):
        vector=cols[:,i].reshape(-1,1)
        matrix=vector@ np.transpose(vector)
        rank1_matrixlist.append(matrix)
    
    #function returning the distance between second_order_kstats-l_1*M_1-...-l_{J-1}*M_{J-1} and v\otimes v
    # input is the vector (l|v)

    def findlastcolumn(second_order_kstats):
        def sumofsquare(x):
            l=x[:J]
            v=x[J:]
            v=v.reshape(-1,1)
            n=len(l)
            N=np.zeros((I,I))
            for i in range(n-1):
                N=N+l[i]*rank1_matrixlist[i]
            M=second_order_kstats-N-l[-1]*v@np.transpose(v)
            return np.sum(M*M)
        
        #initialize a random start point
        l0= abs(np.random.randn(J).reshape(-1,1))
        v0 = np.transpose(uniform_direction(I).rvs(1))
        x=np.vstack((l0,v0)).reshape(-1)
        # optimizition
        es = minimize(sumofsquare, x, method='Powell',tol=1e-12,options={'maxiter':500})
        # solve equation
        # recover last column of A
        v=es.x[J:]/np.linalg.norm(es.x[J:])
        return v
    v=findlastcolumn(second_order_kstats)
    #recover A up to permutation and sign
    estimateA=np.hstack((cols,v.reshape(-1,1)))
    return estimateA


# input two matrices $A,B$ of the same size 
# use a greedy algorithm to find the closest matrix to A obtained from B by permuting and changing the sign of columns
# will also return several several similarity measures 
def similarity_measures(B,A):
    J=A.shape[1]
    columnlist=[]
    last_column_cosine_similarity=np.abs(np.sum(B[:,-1]*A[:,-1]))
    myvector=B[:,-1]
    if last_column_cosine_similarity<0:
        myvector=B[:,-1]
    for i in range(J-1):
        v=A[:,i]
        dislist=[]
        signlist=[]
        for j in range(B.shape[1]-1):
            u=abs(v-B[:,j])
            uprime=abs(v+B[:,j])
            sign=1
            if np.sum(u)<np.sum(uprime):
                dislist.append(np.sum(u))
                signlist.append(sign)
            else:
                sign=-1
                dislist.append(np.sum(uprime))
                signlist.append(sign)
        a=min(dislist)
        index=dislist.index(a)
        sign=signlist[index]
        columnlist.append(sign*B[:,index])
        B=np.delete(B,index,1)
    columnlist.append(myvector)
    permutedB=np.transpose(np.vstack(tuple(columnlist)))
    C= permutedB-A
    relfroberror=(np.sum(C*C)/J)**0.5
    froberror=(np.sum(C*C))**0.5
    cosine_similarity=np.mean(np.sum(permutedB*A,axis=0))
    # for now the bound is 0.95
    if last_column_cosine_similarity>=0.95:
        succesfulflag_last_column=1
    else:
        succesfulflag_last_column=0
    return last_column_cosine_similarity,cosine_similarity,relfroberror,froberror,succesfulflag_last_column,permutedB,A


# function to calculate the second and fourth cumulant tensors from observed data
def cumulant_tensors(observeddata):
    I=observeddata.shape[1]
    sym_indices,b,c=helper_functions.symmetric_indices(I,4)
    fourth_cumulants=np.apply_along_axis(lambda x: kstat(observeddata,tuple(x)), 0, sym_indices)
    fourth_cumulant_dict={tuple(sym_indices[:,n]):fourth_cumulants[n] for n in range(len(fourth_cumulants))}
    all_indices=np.array([list(i) for i in product(range(I), range(I),range(I),range(I))])
    values=np.apply_along_axis(lambda x:fourth_cumulant_dict[tuple(np.sort(x))],1,all_indices)
    fourth_order_kstats=values.reshape(I,I,I,I)
    
    sym_indices_2,b,c=helper_functions.symmetric_indices(I,2)
    second_cumulants=np.apply_along_axis(lambda x: kstat(observeddata,tuple(x)), 0, sym_indices_2)
    second_cumulant_dict={tuple(sym_indices_2[:,n]):second_cumulants[n] for n in range(len(second_cumulants))}
    all_indices_2=np.array([list(i) for i in product(range(I), range(I))])
    values=np.apply_along_axis(lambda x:second_cumulant_dict[tuple(np.sort(x))],1,all_indices_2)
    second_order_kstats=values.reshape(I,I)
    return second_order_kstats,fourth_order_kstats


# generate simulated data
# generate the sources, all but the last one are :
# exponential distribution with parameter one(flag=0) and the last one is standard Gaussian
# student-t distribution with parameter five(flag=1) and the last one is standard Gaussian
def observeddata(samplesize,I,J,flag):
    if flag==1:
        sourcelist=[]
        for i in range(J-1):
            #sourcelist.append(np.random.exponential(1,samplesize))
            sourcelist.append(np.random.standard_t(5,samplesize))
        sourcelist.append(np.random.normal(0,1,samplesize))
        sourcedata=np.transpose(np.vstack(tuple(sourcelist)))

        # generate a random mixing matrix with columns having norm 1
        uniformsphere=uniform_direction(I)
        A=np.transpose(uniformsphere.rvs(J))

        # mix the sources
        observeddata=np.transpose(A @ np.transpose(sourcedata))
    else:
        sourcelist=[]
        for i in range(J-1):
            #sourcelist.append(np.random.exponential(1,samplesize))
            sourcelist.append(np.random.exponential(1,samplesize))
        sourcelist.append(np.random.normal(0,1,samplesize))
        sourcedata=np.transpose(np.vstack(tuple(sourcelist)))

        # generate a random mixing matrix with columns having norm 1
        uniformsphere=uniform_direction(I)
        A=np.transpose(uniformsphere.rvs(J))

        # mix the sources
        observeddata=np.transpose(A @ np.transpose(sourcedata))
    return A,observeddata


# recover mixing matrix using real data or synthetic data
def recovermixingmatrix_sample(I,J,observeddata):
    # generate the second and fourth cumulant tensors
    second_order_kstats,fourth_order_kstats=cumulant_tensors(observeddata)
    # recover first J-1 columns via tensor decomposition
    cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
    def returnmindistancebewteenvectors(cols):
        def distancebewteenvectors(v1,v2):
            v1=v1.reshape(-1,1)
            v2=v2.reshape(-1,1)
            M=v1@np.transpose(v1)-v2@np.transpose(v2)
            return np.sum(M*M)
        lens=cols.shape[1]
        error=1
        for i in range(lens):
            for j in range(i+1,lens):
                error=min(error,distancebewteenvectors(cols[:,i],cols[:,j]))
        return error
    step=0
    while returnmindistancebewteenvectors(cols)<0.1 and step<100:
        cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
        step+=1
    
    # now we use second cumulant to find the last column of the mixing matrix 
    # by minimizing the distance bewteen a matrix in the linear span of the second cumulant matrix 
    # and all the rank 1 matrices M_1,...M_{J-1} obtained from the first J-1 columns of A
    rank1_matrixlist=[]
    for i in range(J-1):
        vector=cols[:,i].reshape(-1,1)
        matrix=vector@ np.transpose(vector)
        rank1_matrixlist.append(matrix)
    
    #function returning the distance between second_order_kstats-l_1*M_1-...-l_{J-1}*M_{J-1} and v\otimes v
    # input is the vector (l|v)
    def sumofsquare(x):
        l=x[:J]
        v=x[J:]
        v=v.reshape(-1,1)
        n=len(l)
        M=second_order_kstats
        for i in range(n-1):
            M=M-l[i]*rank1_matrixlist[i]
        M=M-l[-1]*v@np.transpose(v)
        return np.sum(M*M)
    #initialize a random start point
    l0= np.random.randn(J).reshape(-1,1)
    v0 = np.transpose(uniform_direction(I).rvs(1))
    x=np.vstack((l0,v0)).reshape(-1)
    # optimizition
    es = minimize(sumofsquare, x, method='Powell')
    # recover last column of A
    v=es.x[J:]/np.linalg.norm(es.x[J:])
    #recover A up to permutation and sign
    estimateA=np.hstack((cols,v.reshape(-1,1)))
    return estimateA




# recover mixing matrix using real data or synthetic data with better optimization
def recovermixingmatrix_realdata(I,J,observeddata):
    # generate the second and fourth cumulant tensors
    second_order_kstats,fourth_order_kstats=cumulant_tensors(observeddata)
    # recover first J-1 columns via tensor decomposition
    cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
    def returnmindistancebewteenvectors(cols):
        def distancebewteenvectors(v1,v2):
            v1=v1.reshape(-1,1)
            v2=v2.reshape(-1,1)
            M=v1@np.transpose(v1)-v2@np.transpose(v2)
            return np.sum(M*M)
        lens=cols.shape[1]
        error=1
        for i in range(lens):
            for j in range(i+1,lens):
                error=min(error,distancebewteenvectors(cols[:,i],cols[:,j]))
        return error
    step=0
    while returnmindistancebewteenvectors(cols)<0.1 and step<100:
        cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
        step+=1
    
    # now we use second cumulant to find the last column of the mixing matrix 
    # by minimizing the distance bewteen a matrix in the linear span of the second cumulant matrix 
    # and all the rank 1 matrices M_1,...M_{J-1} obtained from the first J-1 columns of A
    rank1_matrixlist=[]
    for i in range(J-1):
        vector=cols[:,i].reshape(-1,1)
        matrix=vector@ np.transpose(vector)
        rank1_matrixlist.append(matrix)
    
    #function returning the distance between second_order_kstats-l_1*M_1-...-l_{J-1}*M_{J-1} and v\otimes v
    # input is the vector (l|v)
    def sumofsquare(x):
        l=x[:J]
        v=x[J:]
        v=v.reshape(-1,1)
        n=len(l)
        M=second_order_kstats
        for i in range(n-1):
            M=M-l[i]*rank1_matrixlist[i]
        M=M-l[-1]*v@np.transpose(v)
        return np.sum(M*M)
    vlist=[]
    errorlist=[]
    for x in range(10):
        #initialize a random start point
        l0= np.random.randn(J).reshape(-1,1)
        v0 = np.transpose(uniform_direction(I).rvs(1))
        x=np.vstack((l0,v0)).reshape(-1)
        # optimizition
        es = minimize(sumofsquare, x, method='Powell')
        # recover last column of A
        v=es.x[J:]/np.linalg.norm(es.x[J:])
        vlist.append(v)
        errorlist.append(es.fun)
    minerrorindex=errorlist.index(min(errorlist))
    v=vlist[minerrorindex]
    #recover A up to permutation and sign
    estimateA=np.hstack((cols,v.reshape(-1,1)))
    return estimateA






# recover mixing matrix from cumulant tensors
def recoveringmatrix_cumulant(J,second_order_kstats,fourth_order_kstats):
    I=second_order_kstats.shape[0]
    cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
    def returnmindistancebewteenvectors(cols):
        def distancebewteenvectors(v1,v2):
            v1=v1.reshape(-1,1)
            v2=v2.reshape(-1,1)
            M=v1@np.transpose(v1)-v2@np.transpose(v2)
            return np.sum(M*M)
        lens=cols.shape[1]
        error=1
        for i in range(lens):
            for j in range(i+1,lens):
                error=min(error,distancebewteenvectors(cols[:,i],cols[:,j]))
        return error
    step=0
    while returnmindistancebewteenvectors(cols)<0.1 and step<100:
        cols,lambdas=subspace_power_method(fourth_order_kstats,n=4,d=I,r=J-1)
        step+=1
    
    # now we use second cumulant to find the last column of the mixing matrix 
    # by minimizing the distance bewteen a matrix in the linear span of the second cumulant matrix 
    # and all the rank 1 matrices M_1,...M_{J-1} obtained from the first J-1 columns of A
    rank1_matrixlist=[]
    for i in range(J-1):
        vector=cols[:,i].reshape(-1,1)
        matrix=vector@ np.transpose(vector)
        rank1_matrixlist.append(matrix)
    #function returning the distance between second_order_kstats-l_1*M_1-...-l_{J-1}*M_{J-1} and v\otimes v
    # input is the vector (l|v)
    def sumofsquare(x):
        l=x[:J]
        v=x[J:]
        v=v.reshape(-1,1)
        n=len(l)
        M=second_order_kstats
        for i in range(n-1):
            M=M-l[i]*rank1_matrixlist[i]
        M=M-l[-1]*v@np.transpose(v)
        return np.sum(M*M)
    vlist=[]
    errorlist=[]
    for x in range(10):
        #initialize a random start point
        l0= np.random.randn(J).reshape(-1,1)
        v0 = np.transpose(uniform_direction(I).rvs(1))
        x=np.vstack((l0,v0)).reshape(-1)
        # optimizition
        es = minimize(sumofsquare, x, method='Powell')
        # recover last column of A
        v=es.x[J:]/np.linalg.norm(es.x[J:])
        vlist.append(v)
        errorlist.append(es.fun)
    minerrorindex=errorlist.index(min(errorlist))
    v=vlist[minerrorindex]
    #recover A up to permutation and sign
    estimateA=np.hstack((cols,v.reshape(-1,1)))
    return estimateA
    
