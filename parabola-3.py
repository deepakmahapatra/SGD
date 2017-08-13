import scipy
import numpy
import numpy.linalg
import scipy.sparse
class Parabola:
	"""Defines a function that looks like: sum_i alpha_i( x_i - c_i)^2
	In other words, a parabula in arbitrary n dimensions"""
	
	def __init__( self, alpha=scipy.ones(2), center=scipy.ones(2) ):
		"""Initialize.
		alpha -- an n - dimensional array defining the alpha_i  coefficients
		center -- an n - dimensional array defining the c_i constants """
		self.alpha = alpha
		self.center = center
		
	def feval( self, X ):
		"""Evaluate the function at x"""
		a = numpy.square(X - self.center)
		ans = numpy.dot( a , self.alpha )
		return ans

	def seval( self, X, ndata=None ):
		"""Stochastic evaluation of the function at x."""
		return self.feval( self, x )
		
	def grad( self, X ):
		"""Evaluate the gradient at x"""
		ans = 2 * scipy.multiply( X - self.center, self.alpha )
		return ans
	
	def boundgrad( self, gradX, bound ):
		norms = numpy.linalg.norm ( gradX, axis=gradX.ndim-1 )
		bndX = gradX
		condition = numpy.linalg.norm ( gradX, axis=gradX.ndim-1 ) > bound
		if gradX.ndim > 1:
			bndX[condition] = gradX[condition] / norms[condition].reshape( gradX[condition].shape[0], 1 )
		else :
			bndX[condition] = gradX[condition] / norms[condition]
		return bndX

	def sgrad( self, X, ndata=None ):
		"""Return a stochastic gradient at x. Returns the gradient of a uniformly random summand"""
		random_points = scipy.random.randint( low=0, high=X.shape[X.ndim-1], size=X.shape[0] )
		rows = numpy.arange( X.shape[0] )
		ans = numpy.zeros_like( X )
		ans[rows, random_points] = 2 * self.alpha.shape[0] * scipy.take(self.alpha,random_points) * (X[rows, random_points] - scipy.take(self.center,random_points))
		ans = self.boundgrad( ans, 1)
		return ans

class ParabolaDir( Parabola ):
	"""Derived class from class Parabola"""
	def sgrad( self, X, ndata=None ):
		"""Return a stochastic gradient at x. Projects the gradient in a uniformly random direction"""
		Nrandom = scipy.random.standard_normal( X.shape ) 
		norms = numpy.linalg.norm ( Nrandom, axis=X.ndim-1 )
		if X.ndim > 1:
			Nrandom = Nrandom / norms.reshape( X.shape[0], 1 )
		else :
			Nrandom = Nrandom / norms
		grad = self.boundgrad(self.grad(X), 1)
		ans = scipy.multiply( scipy.multiply( Nrandom, grad ), Nrandom)
		return ans
