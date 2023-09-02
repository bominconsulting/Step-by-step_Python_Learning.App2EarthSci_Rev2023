# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 21:46:41 2023

@author: ryujih
"""

def dn2rad(chn,dn):
    gain=[0.363545805215835, 0.343625485897064, 0.154856294393539, 
          0.045724172145128, 0.034687809646130, 0.049800798296928, 
          -0.00108296517282724, -0.01089146733283990, 
          -0.00818779878318309, -0.00969827175140380, 
          -0.01448065508157010, -0.01784354634582990, 
          -0.01981969550251960, -0.02167448587715620, 
          -0.02337997220456600, -0.02430375665426250]
    
    offset=[-7.270904541015620, -6.872497558593750, -6.194244384765620, 
            -3.657928466796870, -1.387512207031250, -0.996017456054687, 
             17.6999874114990,  44.1777038574218, 
             66.7480773925781,  79.0608520507812, 
            118.0509033203120, 145.4648742675780, 
            161.5801391601560, 176.7134399414060, 
            190.6496276855460, 198.2243652343750 ]
    
    rad = dn * gain[chn] + offset[chn]
    
    return rad