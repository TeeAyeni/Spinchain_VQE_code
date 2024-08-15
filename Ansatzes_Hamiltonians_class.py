# This file contains definitions for 
# Hamiltonians: XXZ, XY
# Quantum circuits with different number-conserving gates (equivalently spin conserving)


# import necessary classes
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit import ParameterVector, Parameter
import numpy as np
from numpy import kron
from qiskit.circuit.library import RYGate, RZGate  
from qiskit import QuantumCircuit
from qiskit.algorithms import NumPyMinimumEigensolver
from math import factorial

# classes of Hamiltonians
class XXZ:
    def __init__(self, num_sites, amp):
        self.num_sites = num_sites
        self.amp = amp  # the interaction strength amplitude
    
    # function to rotate a list
    def rotate_list(self, my_list):
        x = my_list.pop()
        my_list.insert(0, x)
        return my_list

    # tensor product of the list
    def tensor_the_list(self, list_x):
        H_x = list_x[0]
        for i in range(1, self.num_sites):
            H_x = H_x^list_x[i]
        return H_x
        
    # the function that creates the actual Hamiltonian operator
    def Hamiltonian(self):
        num_sites = self.num_sites
        amp = self.amp

        # first define the Paulis
        X = Pauli('X')
        Y = Pauli('Y')
        Z = Pauli('Z')
        I = Pauli('I')
                
        # The lists below define lists of the first terms in the XXZ Hamiltonian that we want to create: 
        # e.g. the first term involving X is X_1 x X_2 x I_3 x ... x I_n with X on sites 1 and 2
        list_x = [X,X] + [I for i in range(num_sites-2)]
        list_y = [Y,Y] + [I for i in range(num_sites-2)]
        list_z = [Z,Z] + [I for i in range(num_sites-2)]                
        
        # make the Hamiltonian using the lists above: l_X, l_Y, l_Z     
        xx = self.tensor_the_list(list_x).to_label()
        yy = self.tensor_the_list(list_y).to_label()
        zz = self.tensor_the_list(list_z).to_label()
        H = [(xx, 1), (yy, 1), (zz, amp)]
     
        # add the next terms of the Hamiltonian
        for i in range(2,num_sites):
            # rotate the list to advance the interaction terms to new sites
            list_x = self.rotate_list(list_x)
            list_y = self.rotate_list(list_y)
            list_z = self.rotate_list(list_z)
            
            # make tensor product and add the terms. Update the Hamiltonian
            xx = self.tensor_the_list(list_x).to_label()
            yy = self.tensor_the_list(list_y).to_label()
            zz = self.tensor_the_list(list_z).to_label()
            #H = [(xx, 1), (yy,1), (zz, gamma)]
        
            H.append((xx, 1))
            H.append((yy, 1))
            H.append((zz, amp))
                                    
        return H
        
    # method to get the exact ground state energy
    def get_minimum_eigenvalue(self, H_matrix):
        # Use only for small system sizes to get the reference ground state energy
        ref_value = np.real(np.linalg.eigvals(H_matrix).min())        
        return ref_value


class HCBH_ladder_gauged:
    def __init__(self, num_sites, alpha_gauge) -> None:
        self.dim_x = int(num_sites/2)
        self.num_sites = num_sites
        self.alpha_gauge = alpha_gauge
        
        # interaction strengths
        self.J_1 = 1;  # lower leg
        self.J_2 = 1;  # upper leg
        self.J_v = 1   # rungs

    def string_the_list(self, I_list):
        st = ''
        for ch in I_list:
            st = st + ch
        return st

    def Hamiltonian(self):
        L = self.num_sites
        L_x = self.dim_x
        J_1 = self.J_1; J_2 = self.J_2; J_v = self.J_v; 
        alpha_gauge = self.alpha_gauge

        # start with an empty string of I's
        I_g = ['I' for i in range(L)]
        H = []

        # Bond interactions on lower and upper legs
        for i in range(L_x-1):
            ## XX and YY bond interaction terms on lower leg
            # Note: the gauge field on the lower leg is zero and hence does not have any effect 
            xix = I_g.copy(); yiy = I_g.copy()
            xix[2*i] = 'X'; xix[2*i+2] = 'X'
            yiy[2*i] = 'Y'; yiy[2*i+2] = 'Y'
            xix = self.string_the_list(xix)
            yiy = self.string_the_list(yiy)
            int_val = -J_1/2  
            H.append((xix, int_val))
            H.append((yiy, int_val))
        
            ## bond interaction terms on upper leg
            # The gauge used is nonzero only for the upper leg
            g_val = -2*np.pi*alpha_gauge*(-1)**i

            # the "dot product" between X and Y on sites i,j
            xix = I_g.copy();  yiy = I_g.copy()
            xix[2*i+1] = 'X'; xix[2*i+3] = 'X'
            yiy[2*i+1] = 'Y'; yiy[2*i+3] = 'Y'
            xix = self.string_the_list(xix)
            yiy = self.string_the_list(yiy)
            int_val = -np.cos(g_val)*J_2/2   
            H.append((xix, int_val))
            H.append((yiy, int_val))

            # the "cross product" between X and Y on sites i,j
            xiy = I_g.copy();  yix = I_g.copy()
            xiy[2*i+1] = 'X'; xiy[2*i+3] = 'Y'
            yix[2*i+1] = 'Y'; yix[2*i+3] = 'X'
            xiy = self.string_the_list(xiy)
            yix = self.string_the_list(yix)
            int_val = np.sin(g_val)*J_2/2  
            H.append((xiy, int_val))
            H.append((yix, -int_val))

        # rung terms 
        # gauge field is also zero on the rungs 
        for i in range(L_x):
            xx = I_g.copy(); yy = I_g.copy()
            xx[2*i] = 'X'; xx[2*i+1] = 'X'
            yy[2*i] = 'Y'; yy[2*i+1] = 'Y'
            xx = self.string_the_list(xx)
            yy = self.string_the_list(yy)
            int_val = -J_v/2
            H.append((xx, int_val))
            H.append((yy, int_val))

        return H

    # @@@ not best to duplicate this function across classes. Improve!
    def get_minimum_eigenvalue(self, H_matrix):
        # Use only for small system sizes to get the reference ground state energy
        ref_value = np.real(np.linalg.eigvals(H_matrix).min())        
        return ref_value

    
# the circuit ansatze class
class AnsatzCircuit(QuantumCircuit): 
    def __init__(self, 
                 num_particles, 
                 num_sites, 
                 num_params_per_gate,
                 gate_class,
                 entangle_initial_state, 
                 circ_type                           
                ):
        super().__init__(num_sites)
        self.num_parts = num_particles
        self.num_sites = num_sites
        self.num_params_per_gate = num_params_per_gate
        self.gate_class = gate_class
        self.entangle_initial_state = entangle_initial_state           
        self.circ_type = circ_type
        if circ_type == 'BWC':
            self.get_circuit_BWC()
        elif circ_type=='LC':
            self.get_circuit_LC()
        else:
            ValueError('Circuit structure requested for is not available')


# definitions of the gates                                    
    def A_gate(self, theta, phi):
        # Particle-conserving A-gate with 2 parameters: 
        qc = QuantumCircuit(2)
        
        # "Fusion" gate to enter symmetry basis
        qc.cx(1,0)     
        
        # the controlled-V gate
        qc.rz(-phi,1)            
        qc.ry(-theta,1)             
        qc.cx(0,1)               
        qc.ry(theta,1)           
        qc.rz(phi,1)             
                
        # finally, apply the "splitting" gate to exit symmetry basis
        qc.cx(1,0)     
        
        qc_gate = qc.to_gate(label="A")
        return qc_gate    

    def B_gate(self, theta, phi):
        qc = QuantumCircuit(2)
        qc.cx(1,0)
        qc.x(0)
        qc.cp(phi, 0, 1)
        qc.x(0)
        qc.crx(2*theta, 0, 1)
        qc.cx(1,0)
        #print(qc.draw())                
        qc_gate = qc.to_gate(label="B")        
        return qc_gate


    def G_gate(self, phi1, theta, phi2, alpha):
        # Generalized particle-conserving A-gate with 3+1 parameters: 3 variational and 1 index parameter.
        # the variational parameters are local arguments to the gate, while the index parameter is set uniformly
        # which is set below with "alpha", the phase angle.
        
        qc = QuantumCircuit(2)
        
        # "Fusion" gate to enter symmetry basis
        qc.cx(1,0)     
        
        # the controlled-U gate
        qc.rz((phi2-phi1)/2,1)     # C
        qc.cx(0,1)                 # X
        qc.rz(-(phi1+phi2)/2,1)    # B
        qc.ry(-theta,1)            # B   
        qc.cx(0,1)                 # X
        qc.ry(theta,1)             # A
        qc.rz(phi1,1)              # A
        
        qc.p(alpha, 0)            #apply "phase" gate
                
        # finally, apply the "splitting" gate to exit symmetry basis
        qc.cx(1,0)     
        
        qc_gate = qc.to_gate(label="G")
        return qc_gate
        
    
    def any_gate(self):
        # the general two-qubit gate that do not necessarily conserves particle number
        pass  # insert here the general implementation of the two-qubit decomposition
          
        
    def generate_sites_for_initial_state(self, num_sites):
        # we generate the sites to place the x gates for the case of num_sites odd
        # The case of even num of sites can be obtained easily by shiting 
    
        from math import floor
    
        midpoint = int((num_sites+1)/2)
        
        even_steps=[]
        for k in range(1, floor((num_sites-1)/4)+1):
            even_steps.append(int(midpoint+2*k))
            even_steps.append(int(midpoint-2*k))
    
        odd_steps=[]
        for k in range(1, floor((num_sites+1)/4)+1):
            odd_steps.append(int(midpoint+(2*k-1)))
            odd_steps.append(int(midpoint-(2*k-1)))        
    
        if midpoint%2==0:    
            even_steps.insert(0,midpoint)
            return even_steps, odd_steps
        else:       
            # if midpoint is odd, all the steppings with 2 are odd numbers
            temp=even_steps; even_steps=odd_steps; odd_steps=temp # swap        
            odd_steps.insert(0,midpoint)    
            return odd_steps, even_steps
    
        # notice the order of return. if midpoint is odd, then return (Os,Es), else return (Es,Os)
    
    
    def create_initial_state(self):
        # the two options are either initial product state or entangled state at theta = pi/4, phi's=0 
        # i.e. the gates with determinant = 1
        
        # start with product state and later apply entangling gate if needed
        num_sites = self.num_sites
        num_parts = self.num_parts 
        circ_type = self.circ_type                                    
        
        # -- first, initialize the processor with the number of particles --
        if num_parts <= num_sites:
            
            # We place the initial x gates in a manner that maximizes the spacings between them
            # First, get the sites to place the particles alternately
            if num_sites%2==1:
                S1, S2 = self.generate_sites_for_initial_state(num_sites)
            else:
                S1, S2 = self.generate_sites_for_initial_state(num_sites+1)   # make odd
                # bring back down
                S1 = [x-1 for x in S1]
                S2 = [x-1 for x in S2]
                # delete the 0
                if 0 in S1:
                    S1.pop()
                else:
                    S2.pop()
            
            # shift, for zero-based indexing
            S1 = [x-1 for x in S1]
            S2 = [x-1 for x in S2]
                
            # now place the x gates            
            for i in range(0,num_parts):
                if i<len(S1):
                    pos = S1[i]
                    self.x(pos)
                else:
                    pos = S2[i - len(S1)]
                    self.x(pos)
            
        else:
            raise ValueError("N has to be less or equal to L")
        
        
        # product state
        if self.entangle_initial_state=="no":
            return # initial state is already created                                
        
        elif self.entangle_initial_state=="yes":
            if circ_type =='BWC':
                # apply the entangling gate B at theta=pi/4, alpha=0
                even_bonds = [(i,i+1) for i in range(num_sites) if i%2==0 and i<num_sites-1]
                odd_bonds = [(i,i+1) for i in range(num_sites) if i%2==1 and i<num_sites-1]
            
                # place gates on even sites
                for es in even_bonds:
                    gate = self.B_gate(np.pi/4, 0) 
                    self.append(gate, [es[0], es[1]])    
                # and on odd sites                    
                for os in odd_bonds:
                    gate = self.B_gate(np.pi/4, 0)
                    self.append(gate, [os[0], os[1]])  

            elif circ_type=='LC':
                bonds = [(i,i+1) for i in range(num_sites) if i<num_sites-1]            
            
                # place gates on even sites
                for bond in bonds:
                    gate = self.B_gate(np.pi/4, 0) 
                    self.append(gate, [bond[0], bond[1]])             

                
    
    def get_circuit_BWC(self):               
        # calculate the dimension of (Fock) space
        num_sites = self.num_sites
        num_parts = self.num_parts        
        dim_subspace = int(factorial(num_sites)/(factorial(num_sites - num_parts)*factorial(num_parts)))
        self.dim_subspace = dim_subspace                                   
    
        # create initial state
        self.create_initial_state()
       
        
        # -- second, create the brickwall circuit: apply gates on odd and even bonds --                
        
        # calculate the number of gates needed. This depends on the number of parameters per gate
        num_params_per_gate = self.num_params_per_gate
        if num_params_per_gate==1:
            num_gates = 2*dim_subspace
            self.num_gates = num_gates
        elif num_params_per_gate>=2:
            num_gates = dim_subspace
            self.num_gates = num_gates
        else: 
            raise ValueError("Wrong number of parameters per gate specified")
                
        # create the parameter vector
        theta = ParameterVector(name='θ', length=num_gates)
        phi1 = ParameterVector(name='𝜙1', length=num_gates)              
        phi2 = ParameterVector(name='𝜙2', length=num_gates)              
        phi = ParameterVector(name='𝜙', length=num_gates)                         
        alpha = ParameterVector(name='𝛼', length=num_gates)

        
        # generate the indices for the even and odd bonds
        even_bonds = [(i,i+1) for i in range(0, num_sites) if i%2==0 and i<num_sites-1]
        odd_bonds = [(i,i+1) for i in range(0, num_sites) if i%2==1 and i<num_sites-1]
        
        # place the local gates one by one into the circuit        
        i = 0
        while i < num_gates:
            
            # place gates on even sites
            for es in even_bonds:
                if i < num_gates:                    
                    if self.gate_class=="G":
                        gate = self.G_gate(phi1[i], theta[i], phi2[i], alpha[i])
                    elif self.gate_class=="A":
                        gate = self.A_gate(theta[i], phi[i])                    
                    elif self.gate_class=="B":
                        gate = self.B_gate(theta[i], phi[i])                                            
                    else:
                        raise ValueError('invalid input')   
                
                # now, place gate
                self.append(gate, [es[0], es[1]])
                i += 1                    
            
            # place gates on odd sites
            for os in odd_bonds:
                if i < num_gates:
                    if self.gate_class=="G":
                        gate = self.G_gate(phi1[i], theta[i], phi2[i], alpha[i])
                    elif self.gate_class=="A":
                        gate = self.A_gate(theta[i], phi[i])                    
                    elif self.gate_class=="B":
                        gate = self.B_gate(theta[i], phi[i])                                            
                    else:
                        raise ValueError('invalid input')                          

                # now, place gate
                self.append(gate, [os[0], os[1]])
                i += 1                    

        if self.gate_class=="G":
            self.theta = theta
            self.phi1 = phi1
            self.phi2 = phi2
            self.alpha = alpha            
        elif self.gate_class=="A" or "B":
            self.theta = theta
            self.phi = phi            
        else:
            raise ValueError('invalid input')                 
            
        # Apply the constraints to reduce the number of parameters. Can be used to obtain different circuits
        # Update: This is now done within the caller code


    def get_circuit_LC(self):               
        # calculate the dimension of (Fock) space
        num_sites = self.num_sites
        num_parts = self.num_parts        
        dim_subspace = int(factorial(num_sites)/(factorial(num_sites - num_parts)*factorial(num_parts)))
        self.dim_subspace = dim_subspace                                   
    
        # create initial state
        self.create_initial_state()
       
        
        # -- second, create the brickwall circuit: apply gates on odd and even bonds --                
        
        # calculate the number of gates needed. This depends on the number of parameters per gate
        num_params_per_gate = self.num_params_per_gate
        if num_params_per_gate==1:
            num_gates = 2*dim_subspace
            self.num_gates = num_gates
        elif num_params_per_gate>=2:
            num_gates = dim_subspace
            self.num_gates = num_gates
        else: 
            raise ValueError("Wrong number of parameters per gate specified")
                
        # create the parameter vector
        theta = ParameterVector(name='θ', length=num_gates)
        phi1 = ParameterVector(name='𝜙1', length=num_gates)              
        phi2 = ParameterVector(name='𝜙2', length=num_gates)              
        phi = ParameterVector(name='𝜙', length=num_gates)                         
        alpha = ParameterVector(name='𝛼', length=num_gates)

        
        # generate the indices for the even and odd bonds
        bonds = [(i,i+1) for i in range(num_sites) if i<num_sites-1]
        
        # place the local gates one by one into the circuit        
        i = 0
        while i < num_gates:
            
            # place gates on even sites
            for bond in bonds:
                if i < num_gates:                    
                    if self.gate_class=="G":
                        gate = self.G_gate(phi1[i], theta[i], phi2[i], alpha[i])
                    elif self.gate_class=="A":
                        gate = self.A_gate(theta[i], phi[i])                    
                    elif self.gate_class=="B":
                        gate = self.B_gate(theta[i], phi[i])                                            
                    else:
                        raise ValueError('invalid input')   
                
                # now, place gate
                self.append(gate, [bond[0], bond[1]])
                i += 1                                                           

        if self.gate_class=="G":
            self.theta = theta
            self.phi1 = phi1
            self.phi2 = phi2
            self.alpha = alpha            
        elif self.gate_class=="A" or "B":
            self.theta = theta
            self.phi = phi            
        else:
            raise ValueError('invalid input')                         