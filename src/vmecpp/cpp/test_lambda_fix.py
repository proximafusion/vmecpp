#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import vmecpp

def test_asymmetric_lambda_fix():
    """Test asymmetric mode after lambda coefficient processing fix."""
    
    # Test simple asymmetric case
    config = {
        'delt': 0.5,
        'nfp': 1,
        'ncurr': 1,
        'niter': 100,
        'nstep': 200,
        'nvacskip': 6,
        'ftol_array': [1e-12, 1e-13, 1e-14],
        'ntheta': 16,
        'nzeta': 12,
        'mpol': 6,
        'ntor': 4,
        'lasym': True,
        'rbc': [[1.0, 0.3, 0.0], [0.0, 0.1, 0.0]],
        'zbs': [[0.0, 0.3, 0.0], [0.0, 0.1, 0.0]],
        'rbs': [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],  # Small asymmetric perturbation
        'zbc': [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],  # Small asymmetric perturbation
    }

    try:
        print('=== Testing Asymmetric Mode with Lambda Coefficient Processing ===')
        print('Creating VMEC object...')
        vmec = vmecpp.Vmec(config)
        print('✓ VMEC object created successfully')
        print(f'Configuration lasym: {vmec.indata.lasym}')
        
        print('\nRunning first 5 iterations to test lambda coefficient processing...')
        last_result = None
        for i in range(5):
            try:
                result = vmec.run_vmec_iterations(1)
                last_result = result
                print(f'Iteration {i+1}: ier={result.ier}')
                
                if result.ier != 0:
                    print(f'✗ Failed at iteration {i+1}: ier={result.ier}')
                    return False
                    
            except Exception as e:
                print(f'✗ Exception at iteration {i+1}: {e}')
                return False
        
        print('✓ Asymmetric mode running successfully after lambda coefficient fix!')
        print(f'Final ier: {last_result.ier if last_result else "N/A"}')
        return True
        
    except Exception as e:
        print(f'✗ Error during initialization: {e}')
        return False

if __name__ == '__main__':
    success = test_asymmetric_lambda_fix()
    sys.exit(0 if success else 1)