

## tools for cluster 

def load_cluster(Cluster) :
    return(Cluster.STotElem, Cluster.STot, Cluster.SLevel, Cluster.SizeLevel, Cluster.mesh, Cluster.Ne)


def DimSLevel_def(Slevel, StotElem) :

    DimSLevel = [] 
    AssemSLevel = []

    for i in range(len(Slevel)):

        DimSLevel_i = []
        AssemSLevel_i = []

        for k in range(len(Slevel[i])):

            n = 0
            l = []

            for s_ik in Slevel[i][k] :
                n += len(StotElem[s_ik-1])
                l += StotElem[s_ik-1]

            DimSLevel_i.append( n )
            AssemSLevel_i.append( l )

        DimSLevel.append( DimSLevel_i )
        AssemSLevel.append( AssemSLevel_i )
    
    return(DimSLevel, AssemSLevel)