# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hzKB0QopdzmYrQ6U_cqNO-s2Af2KVSwb
"""



import numpy as np
import qiskit



class ExtendGate():
    
    ################ Weiwen on 06-02-2021 ################
    # Function: ccz from Listing 3
    # Note: using the basic Toffoli gates and CZ gate
    #       to implement ccz gate, which will flip the
    #       sign of state |111>
    # Parameters: (1) quantum circuit; 
    #             (2-3) control qubits;
    #             (4) target qubits;
    #             (5) auxiliary qubits.
    ######################################################
    @classmethod
    def fwd_ccx(cls,circ, q1, q2, q3):
        circ.h(q3)
        circ.cx(q1, q3)
        circ.tdg(q3)
        circ.cx(q2, q3)
        circ.t(q3)
        circ.cx(q1, q3)
        circ.tdg(q3)
        circ.cx(q2, q3)
        circ.t(q2)
        circ.t(q3)
        circ.h(q3)
        circ.swap(q2, q3)
        circ.cx(q1, q2)
        circ.t(q1)
        circ.tdg(q2)
        circ.cx(q1, q2)
        return circ
    @classmethod
    def ori_bridge(cls, circ, q1, q2, q3):
        circ.cx(q2,q3)
        circ.cx(q1,q2)
        circ.cx(q2,q3)
        circ.cx(q1,q2)
        return circ
    @classmethod
    def zero_bridge(cls, circ, q1, q2, q3):
        circ.cx(q1, q2)
        circ.cx(q2, q3)
        circ.cx(q1, q2)
        return circ
    @classmethod
    def my_decompose_cz(cls, circ, q1, q2):
        circ.swap(q1,q2)
        circ.cz(q1, q2)
        circ.swap(q1,q2)
        return circ
    @classmethod
    def my_decompose_cccz(cls, circ, q1, q2, q3, q4, aux1, aux2):
        cls.fwd_ccx(q1, q2, aux1)
        cls.fwd_ccx(q3, aux1, aux2)
        cls.my_decompose_cz(aux2, q4)
        cls.bwd_ccx1(q3, aux1, aux2)
        cls.bwd_ccx2(q1, q2, aux1)
        return circ
    @classmethod
    def ccz(cls,circ, q1, q2, q3, aux1):
        # Apply Z-gate to a state controlled by 3 qubits
        circ.ccx(q1, q2, aux1)
        circ.cz(aux1, q3)
        # cleaning the aux bit
        circ.ccx(q1, q2, aux1)
        return circ
    @classmethod
    def bwd_ccx1(cls, circ, q1, q2, q3):
        circ.swap(q2, q3)
        circ.h(q3)
        circ.cx(q1, q3)
        circ.tdg(q3)
        circ.cx(q2, q3)
        circ.t(q3)
        circ.cx(q1, q3)
        circ.tdg(q3)
        circ.cx(q2, q3)
        circ.t(q2)
        circ.t(q3)
        circ.h(q3)
        cls.zero_bridge(circ,q1, q2, q3)
        circ.cx(q1, q2)
        circ.t(q1)
        circ.tdg(q2)
        circ.cx(q1, q2)
        return circ

    @classmethod
    def cccx(cls,circ, q1, q2, q3, q4, aux1, aux2):
        # Apply Z-gate to a state controlled by 3 qubits
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, aux1, aux2)
        circ.cx(aux2, q4)
        # cleaning the aux bits
        circ.ccx(q3, aux1, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: cccz from Listing 3
    # Note: using the basic Toffoli gates and CZ gate
    #       to implement cccz gate, which will flip the
    #       sign of state |1111>
    # Parameters: (1) quantum circuit; 
    #             (2-4) control qubits;
    #             (5) target qubits;
    #             (6-7) auxiliary qubits.
    ######################################################
    @classmethod
    def cccz(cls,circ, q1, q2, q3, q4, aux1, aux2):
        # Apply Z-gate to a state controlled by 4 qubits
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, aux1, aux2)
        circ.cz(aux2, q4)
        # cleaning the aux bits
        circ.ccx(q3, aux1, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: cccz from Listing 4
    # Note: using the basic Toffoli gate to implement ccccx
    #       gate. It is used to switch the quantum states
    #       of |11110> and |11111>.
    # Parameters: (1) quantum circuit; 
    #             (2-5) control qubits;
    #             (6) target qubits;
    #             (7-8) auxiliary qubits.
    ######################################################
    @classmethod
    def ccccx(cls,circ, q1, q2, q3, q4, q5, aux1, aux2):
        circ.ccx(q1, q2, aux1)
        circ.ccx(q3, q4, aux2)
        circ.ccx(aux2, aux1, q5)
        # cleaning the aux bits
        circ.ccx(q3, q4, aux2)
        circ.ccx(q1, q2, aux1)
        return circ

    ################ Weiwen on 12-30-2020 ################
    # Function: neg_weight_gate from Listing 3
    # Note: adding NOT(X) gate before the qubits associated
    #       with 0 state. For example, if we want to flip 
    #       the sign of |1101>, we add X gate for q2 before
    #       the cccz gate, as follows.
    #       --q3-----|---
    #       --q2----X|X--
    #       --q1-----|---
    #       --q0-----z---
    # Parameters: (1) quantum circuit; 
    #             (2) all qubits, say q0-q3;
    #             (3) the auxiliary qubits used for cccz
    #             (4) states, say 1101
    ######################################################
    @classmethod
    def neg_weight_gate(cls,circ,qubits,aux,state):
        idx = 0
        # The index of qubits are reversed in terms of states.
        # As shown in the above example: we put X at q2 not the third position.
        print("state 1",state)
        state = state[::-1]
        print("state 2",state)
        for idx in range(len(state)):
            if state[idx]=='0':
                circ.x(qubits[idx])
        cls.cccz(circ,qubits[0],qubits[1],qubits[2],qubits[3],aux[0],aux[1])
        for idx in range(len(state)):
            if state[idx]=='0':
                circ.x(qubits[idx])

    @classmethod
    def my_decompose_ccz(cls, circ, q1, q2, q3, aux1):
        cls.fwd_ccx(circ, q1, q2, aux1)
        cls.my_decompose_cz(circ, aux1, q3)
        cls.bwd_ccx2(circ, q1, q2, aux1)
        return circ

    @classmethod
    def bwd_ccx2(cls, circ, q1, q2, q3):
        circ.swap(q2, q3)
        circ.h(q3)
        circ.cx(q1, q3)
        circ.tdg(q3)
        circ.cx(q2, q3)
        circ.t(q3)
        circ.cx(q1, q3)
        circ.tdg(q3)
        circ.cx(q2, q3)
        circ.t(q2)
        circ.t(q3)
        circ.h(q3)
        cls.ori_bridge(circ, q1, q2, q3)
        circ.cx(q1, q2)
        circ.t(q1)
        circ.tdg(q2)
        circ.cx(q1, q2)
        return circ
