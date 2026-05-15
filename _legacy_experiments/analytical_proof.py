import sympy as sp

# ==============================================================================
# ⚛️ QuantumNEAT: Analytical Proof of the AI's Strategy
# ==============================================================================
# We use SymPy to analytically analyze the skeleton matrix found by the AI.
# We will prove WHY the AI decided to set P_11 and P_22 to absolute zero!
# ==============================================================================

def run_analytical_proof():
    print("📜 Generating Analytical Proof for the AI's Skeleton State 📜\n")
    
    # Define symbolic variables
    p_11, p_22, gamma = sp.symbols('P_11 P_22 gamma', real=True, positive=True)
    
    # In the original state, the AI created coherence between |12> and |21>
    # State block for |12> and |21>:
    print("1. The AI created an off-diagonal term (gamma) between |12> and |21>.")
    
    # When we take the Partial Transpose (PT), Bob's indices flip.
    # |12><21| becomes |11><22|
    # So in the PT matrix, the gamma term moves and connects |11> and |22>.
    print("2. Under Partial Transpose, this term moves to connect |11> and |22>.")
    
    # The 2x2 block in the Partial Transposed matrix is:
    PT_block = sp.Matrix([
        [p_11, -gamma],
        [-gamma, p_22]
    ])
    print("\n3. The 2x2 block in the Partial Transpose matrix is:")
    sp.pprint(PT_block)
    
    # For the state to be NPT (Negative Partial Transpose), this block must have a negative eigenvalue.
    # A 2x2 positive-trace matrix has a negative eigenvalue IF AND ONLY IF its determinant is negative.
    det = PT_block.det()
    print("\n4. For the state to be NPT, the determinant must be < 0.")
    print("Determinant:")
    sp.pprint(det)
    
    print("\n5. Condition for NPT:")
    print(f"{det} < 0  ==>  P_11 * P_22 < gamma^2")
    
    print("\n💡 THE AI'S GENIUS MOVE 💡")
    print("To make the state NPT, the AI had to satisfy P_11 * P_22 < gamma^2.")
    print("Instead of making 'gamma' very large (which makes the state highly distillable),")
    print("the AI realized it could just set P_11 = 0 and P_22 = 0!")
    print("If P_11 = 0 and P_22 = 0, then 0 < gamma^2 is ALWAYS true for any small gamma.")
    print("This perfectly explains the '0.0' holes in the skeleton matrix!")

if __name__ == "__main__":
    run_analytical_proof()
