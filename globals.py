import numpy as np
Nrools = 5


REINFORCEMENT_MAX_ROLL=23
AI_MAX_ROLL=24

FEATURE_NUMBER = 9

#[AA,ATA, absAA, R, AAp, ATAmin, Sa, Sr, absHCA]
#Beta = np.array([ 4.05334977e-29, -1.26539848e-28,  4.05334976e-29,  1.14405140e-30, ##beta that is trained with only j = S 250th iter
#  4.05334976e-29, -1.24485881e-41, -1.32998090e-26,  2.84904058e-25,
#  1.85285121e-30])

##beta that is trained with J = S + g_val (20 value is bigger than 1) first iteration, discount is 0.8 rival max roll 23
#Beta = np.array([-4.10438353e-06, -1.44228904e-05, -4.10438341e-06,  3.37741298e-05,
# -4.10438340e-06, -1.85433368e-16, -7.03902326e-02,  5.07524261e-01,
# -4.84619401e-06])
#
##beta that is trained with J = S + g_val (20 value is bigger than 1) 20th iteration, discount is 0.8 rival max roll 23
#Beta = np.array([-1.87510336e-05, -6.06739201e-05, -1.87510336e-05,  1.43067657e-04,
# -1.87510336e-05, -1.70038408e-17,  5.82968066e-03,  4.01371847e-02,
# -2.08347464e-05])
##beta that is trained with J = S + g_val (20 value is bigger than 1) 70th iteration, discount is 0.8 rival max roll 23
#Beta = np.array([-1.90562730e-05, -6.06476360e-05, -1.90562729e-05,  1.44138589e-04,
# -1.90562729e-05, -1.54418023e-17,  6.54395877e-03,  3.57987155e-02,
# -2.09765501e-05])
##beta that is trained with J = S + g_val (20 value is bigger than 1) 100th iteration, discount is 0.8 rival max roll 23
#Beta = np.array([-1.90562834e-05, -6.06476232e-05, -1.90562834e-05,  1.44138609e-04,
# -1.90562834e-05, -1.54417808e-17,  6.54396948e-03,  3.57986553e-02,
# -2.09765506e-05])
##beta that is trained with J = S + g_val (15 value is bigger than 1) 13th iteration, discount is 0.95 rival max roll 19
#Beta = np.array([-2.80900717e-05, -1.27844026e-04, -2.80900717e-05,  2.55106660e-04, 
# -2.80900717e-05, -1.09506427e-16, -2.43196300e-02,  3.17579337e-01,
# -2.88389105e-05])
##beta that is trained with J = S + g_val (17 value is bigger than 1) 13th iteration, discount is 0.95 rival max roll 19
#Beta = np.array([-2.79861901e-05, -1.25537954e-04, -2.79861903e-05,  2.53983994e-04, 
# -2.79861903e-05, -2.14790331e-17, -2.62304590e-02,  3.31565738e-01,
# -3.15163457e-05])
##beta that is trained with J = S + g_val (17 value is bigger than 1) 73th iteration, discount is 0.95 rival max roll 19
#Beta = np.array([-6.40597714e-05, -2.27423994e-04, -6.40597714e-05,  5.09091255e-04, 
# -6.40597714e-05, -1.00404074e-17,  2.28454098e-02,  1.36565301e-01,
# -6.55799129e-05])

##beta that is trained with J = S + g_val (17 value is bigger than 1) 104th iteration, discount is 0.95 rival max roll 19
#Beta = np.array([-6.63871423e-05, -2.29329146e-04, -6.63871424e-05,  5.19673260e-04,
# -6.63871424e-05, -9.60310132e-18,  2.48328901e-02,  1.29051912e-01,
# -6.67229178e-05])

##beta that is trained with J = S + g_val (105 value is bigger than 1) 1st iteration, discount is 0.95 rival max roll 21
#Beta = np.array([-5.62290010e-05, -1.85303419e-04, -5.62290011e-05,  4.27991388e-04,
# -5.62290011e-05, -6.11568956e-17, -4.60736086e-02,  5.78115936e-01,
# -5.22554281e-05])


##beta that is trained with J = S + g_val (105 value is bigger than 1) 87th iteration, discount is 0.95 rival max roll 21
#Beta = np.array([-4.84092372e-04, -1.64910376e-03, -4.84092372e-04,  3.74797724e-03,
# -4.84092372e-04, -1.11107779e-16,  1.87453906e-01,  9.07034113e-01,
# -4.51743215e-04]
#)

##beta that is trained with J = S + g_val (105 value is bigger than 1) 105th iteration, discount is 0.95 rival max roll 21
#Beta = np.array([-4.88004469e-04, -1.66225408e-03, -4.88004469e-04,  3.77796512e-03,
# -4.88004469e-04, -1.11556273e-16,  1.89568898e-01,  9.09977746e-01,
# -4.55273891e-04])

##beta that is trained with J = S + g_val (203 value is bigger than 1) 1st iteration, discount is 0.95 rival max roll 21
#Beta = np.array([-5.84869155e-05, -1.87827500e-04, -5.84869153e-05,  4.33616792e-04,
# -5.84869153e-05,  3.89905863e-17, -4.93606968e-02,  6.05587988e-01,
# -4.79294347e-05])

##beta that is trained with J = S + g_val (203 value is bigger than 1) 35th iteration, discount is 0.95 rival max roll 21
#Beta = np.array([-7.48730064e-04, -2.56921235e-03, -7.48730063e-04,  5.77014579e-03,
# -7.48730063e-04,  1.09447517e-16,  2.84721101e-01,  1.47365871e+00,
# -6.40327234e-04])

##beta that is trained with J = S + g_val (370 value is bigger than 1) 67th iteration, discount is 0.95 rival max roll 21
Beta = np.array([-7.53027510e-04, -2.64746640e-03, -7.53027511e-04,  5.91592059e-03,
 -7.53027511e-04,  3.27319596e-16,  2.94166431e-01,  1.43873963e+00,
 -7.11348069e-04])
