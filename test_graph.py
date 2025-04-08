import runabb.abb as abb
R = abb.Robot(ip='192.168.125.1')
R.set_cartesian([[264.88, -10.7, 708.8], [0,0,1,0]])