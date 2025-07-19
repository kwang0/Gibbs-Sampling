using ITensors
using ITensorMPS
using CUDA

# Representation of two qubits coarse-grained onto doubled Hilbert space
# Convention is (|up,up>, |up,down>, |down,up>, |down,down>)
function ITensors.space(::SiteType"doubled";
  conserve_qns=false)
  if conserve_qns
    return [QN("Sz",1)=>1,QN("Sz",0)=>2,QN("Sz",-1)=>1]
  end
  return 4
end

ITensors.state(::StateName"UpUp", ::SiteType"doubled") = [1.0, 0, 0, 0]
ITensors.state(::StateName"UpDn", ::SiteType"doubled") = [0, 1.0, 0, 0]
ITensors.state(::StateName"DnUp", ::SiteType"doubled") = [0, 0, 1.0, 0]
ITensors.state(::StateName"DnDn", ::SiteType"doubled") = [0, 0, 0, 1.0]
ITensors.state(::StateName"InfMixed", ::SiteType"doubled") = [1.0/sqrt(2), 0, 0, 1.0/sqrt(2)]

ITensors.op(::OpName"Z1",::SiteType"doubled") =
  [ +1    0    0    0
     0   +1    0    0 
     0    0   -1    0
     0    0    0   -1]

ITensors.op(::OpName"Z2",::SiteType"doubled") =
  [ +1    0    0    0
     0   -1    0    0 
     0    0   +1    0
     0    0    0   -1]

ITensors.op(::OpName"X1",::SiteType"doubled") =
  [  0    0    1    0
     0    0    0    1 
     1    0    0    0
     0    1    0    0]

ITensors.op(::OpName"X2",::SiteType"doubled") =
  [  0    1    0    0
     1    0    0    0 
     0    0    0    1
     0    0    1    0]

ITensors.op(::OpName"Y1",::SiteType"doubled") =
  [  0    0    -1im 0
     0    0    0    -1im 
     1im  0    0    0
     0    1im  0    0]

ITensors.op(::OpName"Y2",::SiteType"doubled") =
  [  0    -1im 0    0
     1im  0    0    0 
     0    0    0    -1im
     0    0    1im  0]

ITensors.op(::OpName"Y1_conj",::SiteType"doubled") =
  [  0    0    1im  0
     0    0    0    1im 
     -1im 0    0    0
     0    -1im 0    0]

ITensors.op(::OpName"Y2_conj",::SiteType"doubled") =
  [  0    1im  0    0
     -1im 0    0    0 
     0    0    0    1im
     0    0    -1im 0]

ITensors.op(::OpName"X1X2_conj+Y1Y2_conj",::SiteType"doubled") =
  [0   0   0   2
   0   0   0   0
   0   0   0   0
   2   0   0   0]

ITensors.op(::OpName"P1up",::SiteType"doubled") =
  [1   0   0   0
   0   1   0   0
   0   0   0   0
   0   0   0   0]

ITensors.op(::OpName"P1dn",::SiteType"doubled") =
  [0   0   0   0 
   0   0   0   0
   0   0   1   0
   0   0   0   1]

ITensors.op(::OpName"P2up",::SiteType"doubled") =
  [1   0   0   0
   0   0   0   0
   0   0   1   0
   0   0   0   0]

ITensors.op(::OpName"P2dn",::SiteType"doubled") =
  [0   0   0   0
   0   1   0   0
   0   0   0   0
   0   0   0   1]

ITensors.op(::OpName"Id",::SiteType"doubled") =
  [1   0   0   0
   0   1   0   0
   0   0   1   0
   0   0   0   1]

function find_coeffs(β)
    alphas = zeros(5, 5)
    hs = zeros(5, 5)

    σ_E = 1.0/β
    omegas = LinRange(-10.0, 10.0, 2001)

    for i in 1:5
        for j in 1:5
            ν1 = 2*(i - 3)
            ν2 = 2*(j - 3)

            
        end
    end
end

function parent_hamiltonian(L, alphas, hs)
    os = OpSum()

    for j in 1:(L-2)
        # (-4,-4)
        os .+= hs[1,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1dn", j + 1, "P1up", j + 2, "P2up", j, "P2dn", j + 1, "P2up", j + 2
        os .+= hs[1,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1dn", j + 1, "P1up", j + 2, "P2dn", j, "P2up", j + 1, "P2dn", j + 2
        os .+= hs[1,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1up", j + 1, "P1dn", j + 2, "P2up", j, "P2dn", j + 1, "P2up", j + 2
        os .+= hs[1,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1up", j + 1, "P1dn", j + 2, "P2dn", j, "P2up", j + 1, "P2dn", j + 2

        # (-4,0)
        os .+= hs[1,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1dn", j + 1, "P1up", j + 2, "P2up", j, "Id", j + 1, "P2dn", j + 2
        os .+= hs[1,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1dn", j + 1, "P1up", j + 2, "P2dn", j, "Id", j + 1, "P2up", j + 2
        os .+= hs[1,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1up", j + 1, "P1dn", j + 2, "P2up", j, "Id", j + 1, "P2dn", j + 2
        os .+= hs[1,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1up", j + 1, "P1dn", j + 2, "P2dn", j, "Id", j + 1, "P2up", j + 2

        # (-4,4)
        os .+= hs[1,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1dn", j + 1, "P1up", j + 2, "P2up", j, "P2up", j + 1, "P2up", j + 2
        os .+= hs[1,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1dn", j + 1, "P1up", j + 2, "P2dn", j, "P2dn", j + 1, "P2dn", j + 2
        os .+= hs[1,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1up", j + 1, "P1dn", j + 2, "P2up", j, "P2up", j + 1, "P2up", j + 2
        os .+= hs[1,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1up", j + 1, "P1dn", j + 2, "P2dn", j, "P2dn", j + 1, "P2dn", j + 2
        
        # (0,-4)
        os .+= hs[3,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "Id", j + 1, "P1dn", j + 2, "P2up", j, "P2dn", j + 1, "P2up", j + 2
        os .+= hs[3,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "Id", j + 1, "P1dn", j + 2, "P2dn", j, "P2up", j + 1, "P2dn", j + 2
        os .+= hs[3,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "Id", j + 1, "P1up", j + 2, "P2up", j, "P2dn", j + 1, "P2up", j + 2
        os .+= hs[3,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "Id", j + 1, "P1up", j + 2, "P2dn", j, "P2up", j + 1, "P2dn", j + 2
        
        # (0,0)
        os .+= hs[3,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "Id", j + 1, "P1dn", j + 2, "P2up", j, "Id", j + 1, "P2dn", j + 2
        os .+= hs[3,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "Id", j + 1, "P1dn", j + 2, "P2dn", j, "Id", j + 1, "P2up", j + 2
        os .+= hs[3,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "Id", j + 1, "P1up", j + 2, "P2up", j, "Id", j + 1, "P2dn", j + 2
        os .+= hs[3,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "Id", j + 1, "P1up", j + 2, "P2dn", j, "Id", j + 1, "P2up", j + 2
        
        # (0,4)
        os .+= hs[3,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "Id", j + 1, "P1dn", j + 2, "P2up", j, "P2up", j + 1, "P2up", j + 2
        os .+= hs[3,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "Id", j + 1, "P1dn", j + 2, "P2dn", j, "P2dn", j + 1, "P2dn", j + 2
        os .+= hs[3,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "Id", j + 1, "P1up", j + 2, "P2up", j, "P2up", j + 1, "P2up", j + 2
        os .+= hs[3,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "Id", j + 1, "P1up", j + 2, "P2dn", j, "P2dn", j + 1, "P2dn", j + 2
        
        # (4,-4)
        os .+= hs[5,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1up", j + 1, "P1up", j + 2, "P2up", j, "P2dn", j + 1, "P2up", j + 2
        os .+= hs[5,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1up", j + 1, "P1up", j + 2, "P2dn", j, "P2up", j + 1, "P2dn", j + 2
        os .+= hs[5,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1dn", j + 1, "P1dn", j + 2, "P2up", j, "P2dn", j + 1, "P2up", j + 2
        os .+= hs[5,1], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1dn", j + 1, "P1dn", j + 2, "P2dn", j, "P2up", j + 1, "P2dn", j + 2
        
        # (4,0)
        os .+= hs[5,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1up", j + 1, "P1up", j + 2, "P2up", j, "Id", j + 1, "P2dn", j + 2
        os .+= hs[5,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1up", j + 1, "P1up", j + 2, "P2dn", j, "Id", j + 1, "P2up", j + 2
        os .+= hs[5,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1dn", j + 1, "P1dn", j + 2, "P2up", j, "Id", j + 1, "P2dn", j + 2
        os .+= hs[5,3], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1dn", j + 1, "P1dn", j + 2, "P2dn", j, "Id", j + 1, "P2up", j + 2
        
        # (4,4)
        os .+= hs[5,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1up", j + 1, "P1up", j + 2, "P2up", j, "P2up", j + 1, "P2up", j + 2
        os .+= hs[5,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1up", j, "P1up", j + 1, "P1up", j + 2, "P2dn", j, "P2dn", j + 1, "P2dn", j + 2
        os .+= hs[5,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1dn", j + 1, "P1dn", j + 2, "P2up", j, "P2up", j + 1, "P2up", j + 2
        os .+= hs[5,5], "X1X2_conj+Y1Y2_conj", j + 1, "P1dn", j, "P1dn", j + 1, "P1dn", j + 2, "P2dn", j, "P2dn", j + 1, "P2dn", j + 2

        # N terms

        # (4,4)
        os .+= -alphas[5,5], "P1up", j, "P1up", j + 1, "P1up", j + 2
        os .+= -alphas[5,5], "P1dn", j, "P1dn", j + 1, "P1dn", j + 2
        os .+= -alphas[5,5], "P2up", j, "P2up", j + 1, "P2up", j + 2
        os .+= -alphas[5,5], "P2dn", j, "P2dn", j + 1, "P2dn", j + 2
        
        # (0,0)
        os .+= -alphas[3,3], "P1up", j, "Id", j + 1, "P1dn", j + 2
        os .+= -alphas[3,3], "P1dn", j, "Id", j + 1, "P1up", j + 2
        os .+= -alphas[3,3], "P2up", j, "Id", j + 1, "P2dn", j + 2
        os .+= -alphas[3,3], "P2dn", j, "Id", j + 1, "P2up", j + 2
        
        # (-4,-4)
        os .+= -alphas[1,1], "P1up", j, "P1dn", j + 1, "P1up", j + 2
        os .+= -alphas[1,1], "P1dn", j, "P1up", j + 1, "P1dn", j + 2
        os .+= -alphas[1,1], "P2up", j, "P2dn", j + 1, "P2up", j + 2
        os .+= -alphas[1,1], "P2dn", j, "P2up", j + 1, "P2dn", j + 2


        # Z contributions
        os .+= hs[3,3], "Z1", j + 1, "Z2", j + 1
        os .+= -alphas[3,3], "Id", j + 1

    end
    
    # Boundary terms

    # (-2,-2)
    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1dn", 2, "P2up", 1, "P2dn", 2
    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1dn", 2, "P2dn", 1, "P2up", 2
    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1up", 2, "P2up", 1, "P2dn", 2
    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1up", 2, "P2dn", 1, "P2up", 2

    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1dn", L-1, "P2up", L, "P2dn", L-1
    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1dn", L-1, "P2dn", L, "P2up", L-1
    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1up", L-1, "P2up", L, "P2dn", L-1
    os .+= hs[2,2], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1up", L-1, "P2dn", L, "P2up", L-1

    # (-2,2)
    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1dn", 2, "P2up", 1, "P2up", 2
    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1dn", 2, "P2dn", 1, "P2dn", 2
    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1up", 2, "P2up", 1, "P2up", 2
    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1up", 2, "P2dn", 1, "P2dn", 2

    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1dn", L-1, "P2up", L, "P2up", L-1
    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1dn", L-1, "P2dn", L, "P2dn", L-1
    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1up", L-1, "P2up", L, "P2up", L-1
    os .+= hs[2,4], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1up", L-1, "P2dn", L, "P2dn", L-1

    # (2,-2)
    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1up", 2, "P2up", 1, "P2dn", 2
    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1up", 2, "P2dn", 1, "P2up", 2
    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1dn", 2, "P2up", 1, "P2dn", 2
    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1dn", 2, "P2dn", 1, "P2up", 2

    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1up", L-1, "P2up", L, "P2dn", L-1
    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1up", L-1, "P2dn", L, "P2up", L-1
    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1dn", L-1, "P2up", L, "P2dn", L-1
    os .+= hs[4,2], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1dn", L-1, "P2dn", L, "P2up", L-1

    # (2,2)
    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1up", 2, "P2up", 1, "P2up", 2
    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", 1, "P1up", 1, "P1up", 2, "P2dn", 1, "P2dn", 2
    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1dn", 2, "P2up", 1, "P2up", 2
    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", 1, "P1dn", 1, "P1dn", 2, "P2dn", 1, "P2dn", 2

    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1up", L-1, "P2up", L, "P2up", L-1
    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", L, "P1up", L, "P1up", L-1, "P2dn", L, "P2dn", L-1
    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1dn", L-1, "P2up", L, "P2up", L-1
    os .+= hs[4,4], "X1X2_conj+Y1Y2_conj", L, "P1dn", L, "P1dn", L-1, "P2dn", L, "P2dn", L-1

    # N terms

        # (2,2)
        os .+= -alphas[4,4], "P1up", 1, "P1up", 2
        os .+= -alphas[4,4], "P1dn", 1, "P1dn", 2
        os .+= -alphas[4,4], "P2up", 1, "P2up", 2
        os .+= -alphas[4,4], "P2dn", 1, "P2dn", 2

        os .+= -alphas[4,4], "P1up", L-1, "P1up", L
        os .+= -alphas[4,4], "P1dn", L-1, "P1dn", L
        os .+= -alphas[4,4], "P2up", L-1, "P2up", L
        os .+= -alphas[4,4], "P2dn", L-1, "P2dn", L

        # (-2,-2)
        os .+= -alphas[2,2], "P1up", 1, "P1dn", 2
        os .+= -alphas[2,2], "P1dn", 1, "P1up", 2
        os .+= -alphas[2,2], "P2up", 1, "P2dn", 2
        os .+= -alphas[2,2], "P2dn", 1, "P2up", 2

        os .+= -alphas[2,2], "P1up", L-1, "P1dn", L
        os .+= -alphas[2,2], "P1dn", L-1, "P1up", L
        os .+= -alphas[2,2], "P2up", L-1, "P2dn", L
        os .+= -alphas[2,2], "P2dn", L-1, "P2up", L

    # Z contributions
    os .+= hs[3,3], "Z1", 1, "Z2", 1
    os .+= hs[3,3], "Z1", L, "Z2", L
    os .+= -alphas[3,3], "Id", 1
    os .+= -alphas[3,3], "Id", L

    return os
end

function main()

end