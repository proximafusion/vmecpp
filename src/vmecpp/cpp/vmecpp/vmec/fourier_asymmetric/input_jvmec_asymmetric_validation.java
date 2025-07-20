// jVMEC input configuration for three-code asymmetric validation
// Generated from VMEC++ production-ready asymmetric configuration
// This configuration shows 100% success rate in VMEC++

// Basic configuration
asymmetric = true;
NFP = 1;
MPOL = 7;
NTOR = 0;
nsValues = new int[]{3, 5};
ftolValues = new double[]{1.0e-4, 1.0e-6};
niterValues = new int[]{50, 100};
delt = 0.9;

// Boundary coefficients from working VMEC++ configuration
rbc[0][0] = 5.91630000e+00;
zbc[0][0] = 4.10500000e-01;

// M=1 coefficients (critical for asymmetric convergence)
rbc[1][0] = 1.91960000e+00;
rbs[1][0] = 2.76100000e-02;  // Input value before M=1 constraint
zbc[1][0] = 5.73020000e-02;  // Input value before M=1 constraint
zbs[1][0] = 3.62230000e+00;

// Higher order coefficients
rbc[2][0] = 3.37360000e-01;
rbs[2][0] = 1.00380000e-01;
zbc[2][0] = 4.66970000e-03;
zbs[2][0] = -1.85110000e-01;

rbc[3][0] = 4.15040000e-02;
rbs[3][0] = -7.18430000e-02;
zbc[3][0] = -3.91550000e-02;
zbs[3][0] = -4.85680000e-03;

rbc[4][0] = -5.82560000e-03;
rbs[4][0] = -1.14230000e-02;
zbc[4][0] = -8.78480000e-03;
zbs[4][0] = 5.92680000e-02;

rbc[5][0] = 1.03740000e-02;
rbs[5][0] = 8.17700000e-03;
zbc[5][0] = 2.11750000e-02;
zbs[5][0] = 4.47700000e-03;

rbc[6][0] = -5.63650000e-03;
rbs[6][0] = -7.61100000e-03;
zbc[6][0] = 2.43900000e-03;
zbs[6][0] = -1.67730000e-02;

// Axis coefficients
raxis[0] = 7.5025;
raxis[1] = 0.47;
zaxis[0] = 0.0;
zaxis[1] = 0.0;

// Physics parameters
gamma = 0.0;
ncurr = 0;
pres_scale = 100000.0;
curtor = 0.0;

// Expected M=1 constraint behavior:
// jVMEC should apply: rbs[1] = zbc[1] = (0.027610 + 0.057302) / 2.0 = 0.042456
// This is a 53.77% change in rbs[1] and 25.91% change in zbc[1]
// Critical validation point: does jVMEC apply this constraint automatically?
