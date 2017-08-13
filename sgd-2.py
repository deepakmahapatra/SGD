#importing the library
import scipy
import numpy as np
import matplotlib
import matplotlib.path
import matplotlib.patches
import matplotlib.pylab
import matplotlib.pyplot
import cProfile
import pstats
import parabola
import numpy.linalg as LA





def step_size( gamma, start=1 ):
	"""Return a function that produces a step size for the stochastic gradient descent. The step 		size for the nth step is of the form
	
	start / n^gammma

	For stochastic gradient descent to work, gamma has to be less tha or equal to 1. The parameter 		start defines the length of the first step.
	Returns a function sfunc such that sfunc(k) returns the length of the kth step"""
	def sfunc( k ):
		"""Returns the length of the kth step"""	
		step = start / (float(k) ** gamma)
		return step
	return sfunc



class SGD:
	"""A class implementing stochastic gradient descent"""
	
	def __init__( self, afunc, x0, sfunc, proj=None, histsize=-1, smallhist=False, ndata=100, keepobj=True):
		"""afunc -- the objective. has afunc.sgrad(x, ndata) returning a stochastic subgradient, afunc.feval(x) returning a function evaluation, and a func.sfeval(x,ndata) returning a stochastic function evaluation
		x0 -- initial point
		sfunc -- a step function. sfunc(n) returns the size of the nth step
		proj --a projection function. proj(x) returns the closest point to x within the feasible region
		histsize -- how many steps of history to maintain (-1 is all the steps)
		smallhist -- whether to maintain the history of gradients, stepsizes, and pre-projection points
		ndata -- the number of data points to pass into sgrad and sfeval
		keepobj -- whether to maintain a history of the objective function value"""
		self.xvalues = []
		self.obj_val = []
		self.sgradVal = []
		self.ss = []
		self.start = x0
		self.obj = afunc
		self.stepfunc = sfunc
		self.soln = 0
		self.setStart( x0 )

	def setStart( self, x0 ):
		"""Set the start point"""
		self.xvalues.append( x0 )
		self.obj_val.append( self.obj.feval(x0) )
		

	def reset( self ):
		"""Reset the history of the optimization. In other words, drop all history and start again from x0 and 1st step"""
		self.xvalues = []
		self.obj_val = []
		self.ss = []
		self.sgradVal = []
		self.setStart( self.start )

	def dostep( self ):
		"""Take a single step of SGD"""
		x_prev = self.xvalues[-1]
		step = self.stepfunc( len(self.xvalues) )
		sgrad = self.obj.sgrad( x_prev ) 
		x_new = x_prev - step*sgrad
		self.xvalues.append( x_new )
		self.obj_val.append( self.obj.feval(x_new) )
		self.ss.append( step )
		self.sgradVal.append ( sgrad )
		print "Iteration {0}, objective_function {1}".format(len(self.xvalues),self.obj_val[-1])

	def nsteps( self, an=1 ):
		"""Take an steps of SGD"""
		for i in range(an):
			self.dostep()

	def getAvgSoln( self, wsize=10 ):
		"""Average the last wsize points and return a solution"""
		avg = np.mean( self.xvalues[-wsize], axis=0 )
		value = self.obj.feval( avg )
		return value

	def getxStarPrev( self, wsize=10, winterval=1):
		backstep = wsize*(winterval+1)
		
		values = self.xvalues[-wsize*(winterval+2):-wsize*(winterval+1)]
		backavg = np.mean( values, axis=0 )
		x_starprev = self.obj.feval( backavg )
		return x_starprev

	def getSoln( self, wsize=10, winterval=1, abstol=0.000006, reltol=0.000006):
		"""Keep performing SGD steps until: afunc.feval(x*_prev) and afunc.feval(x*) is within the specified tolerances
		x* -- is a solution obtained from averaging the last wsize points
		x*_prev -- is a solution obtained by averaging the wsize points that were wsize*(winterval+1) back in history
		Intuitively, this function keeps performing steps until the objective value does not change much."""
		self.nsteps( an=30 )
		
		x_star = self.getAvgSoln()
		x_starprev = self.getxStarPrev( wsize=wsize, winterval=winterval )	
		while ((np.any(np.absolute(x_star - x_starprev)/(x_starprev) > reltol)) and np.any((np.absolute(x_star - x_starprev) > abstol))):
			self.nsteps(an=10)
			x_star = self.getAvgSoln()
			x_starprev = self.getxStarPrev( wsize=wsize, winterval=winterval )	
		self.soln = self.xvalues[-1]			
		print "Final solution {0} *** objective value is {1}".format(self.xvalues[-1], self.obj_val[-1])
	#function for plotting	
	
	def plot( self, fname=None, n=200, alphaMult=0.7, axis=None):
		"""Produce a plot of the last n SGD steps.
		fname -- a file name where to save the plot, or show if None 
		n -- the number of points to display
		alphaMult -- a geometric multiplier on the alpha value of the segments, with the most recent one having alpha=1
		axis -- the axis on which to plot the steps."""	
		t = len(self.xvalues)
		n = min(n, t)
		points = scipy.array(self.xvalues[:n])
		
		colors = ['purple','red','cyan','blue','green','black']
		numberSS = points.shape[1]
		color=colors[:numberSS]
		codes = [matplotlib.patches.Path.MOVETO, matplotlib.patches.Path.LINETO]
		for i in range(points.shape[1]):
			pts = points[:,i,:]
			alpha=0.7
			
			matplotlib.pylab.plot(*zip(*pts), color=colors[i%len(colors)], alpha=alpha)
		#saving the figure as in the required path.	
		if fname is None:
			matplotlib.pylab.show()
		else:
			matplotlib.pylab.savefig(fname)
			


			
		

if __name__ == '__main__':
	a = np.array([10,200,40])
	c = np.array([2,1,4])
	p = parabola.ParabolaDir(alpha=a, center=c)
	
	startx = np.matrix([3,3,3])
	s = SGD(afunc=p, x0=startx, sfunc=step_size(gamma=0.9,start=10))
	
	
	a = np.array([1,1])
	c = np.array([0,0])
	p = parabola.ParabolaDir(alpha=a, center=c)
	startx = np.matrix([3,3])
	s = SGD(afunc=p, x0=startx, sfunc=step_size(gamma=0.9,start=1))
	#Running for the five instances as stated in the homework
	#1
	for i in range(200):
		print i
		s.nsteps(1)
		matplotlib.pylab.clf()
		a=matplotlib.pylab.gca()
		matplotlib.pylab.scatter([0],[0], marker='o', color='red')
		fname = '/Users/deepakmahapatra/Documents/CompOPVideo/HW3/All/parabA1%03d.jpg'%i
		s.plot(fname, alphaMult=0.9, axis=a)
	
	a = np.array([10,200])
	c = np.array([2,1])
	p = parabola.ParabolaDir(alpha=a, center=c)
	startx = np.matrix([3,3])
	#2
	s = SGD(afunc=p, x0=startx, sfunc=step_size(gamma=0.9,start=10))
	for i in range(200):
		s.nsteps(1)
		matplotlib.pylab.clf()
		a=matplotlib.pylab.gca()
		matplotlib.pylab.scatter([2],[1], marker='o', color='red')
		fname = '/Users/deepakmahapatra/Documents/CompOPVideo/HW3/All/parabB2%03d.jpg'%i
		s.plot(fname, alphaMult=0.9, axis=a)
	
	a = np.array([10,200])
	c = np.array([2,1])
	p = parabola.Parabola(alpha=a, center=c)
	startx = np.matrix([3,3])
	#3
	s = SGD(afunc=p, x0=startx, sfunc=step_size(gamma=0.9,start=30))
	for i in range(300):
		s.nsteps(1)
		matplotlib.pylab.clf()
		a=matplotlib.pylab.gca()
		matplotlib.pylab.scatter([2],[1], marker='o', color='red')
		fname = '/Users/deepakmahapatra/Documents/CompOPVideo/HW3/All/parabC3%03d.jpg'%i
		s.plot(fname, alphaMult=0.9, axis=a)
	
	
	a = np.array([10,200,40])
	c = np.array([2,1,4])
	p = parabola.ParabolaDir(alpha=a, center=c)
	startx = np.matrix([3,3,3])
	#4
	s = SGD(afunc=p, x0=startx, sfunc=step_size(gamma=0.9,start=10))
	import mpl_toolkits.mplot3d
	import matplotlib.pyplot
	figure=matplotlib.pyplot.figure()
	for i in range(200):
		s.nsteps(1)
		a=figure.gca(projection='3d')
		a.scatter([2],[1],[4], marker='o', color='red')
		fname = '/Users/deepakmahapatra/Documents/CompOPVideo/HW3/All/parabD4%03d.jpg'%i
		s.plot(fname, alphaMult=0.9, axis=a)
	
	
	a = np.array([10,200])
	c = np.array([2,1])
	p = parabola.ParabolaDir(alpha=a, center=c)
	startx = np.random.uniform(low=-3,high=3,size=(4,2))
	#5
	s = SGD(afunc=p, x0=startx, sfunc=step_size(gamma=0.9,start=10))
	for i in range(200):
		s.nsteps(1)
		matplotlib.pylab.clf()
		a=matplotlib.pylab.gca()
		matplotlib.pylab.scatter([2],[1], marker='o', color='red')
		fname = '/Users/deepakmahapatra/Documents/CompOPVideo/HW3/All/parabE5%03d.jpg'%i
		s.plot(fname, alphaMult=0.9, axis=a)
	
	
