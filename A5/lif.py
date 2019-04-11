from numpy import *
from scipy import signal, misc
import matplotlib.pyplot as plt
import pdb
#pdb.set_trace()  # this sets a breakpoint

def Population_ST(N, xmin, max_freq, dim=100):
	"""
	Blah
	"""

	dt = 0.001 # 1 ms

	myA = GenerateLIFPopulation(N, xmin, max_freq, dim)
	t = arange(dim) - dim + 1
	t = t.astype(float)*dt

	for n in range(N):
		temporal_freq = 50.0 + 30*random.rand()
		temporal_decay = (15.0 + 6*random.randn())*dt
		e = sin(-temporal_freq*t) * exp(t/temporal_decay)
		if random.rand()>0.5:
			myA[5:,n] = e
		else:
			myA[5:,n] = -e

	return myA


def PlotTuningCurves(pop, xlim=[-1,1]):
	P = 200
	x = linspace(xlim[0], xlim[1], P)
	A = Stim2Rate(x, pop)
	plt.plot(x, A)
	plt.xlabel('Input')
	plt.ylabel('Firing Rate (Hz)');

def GenerateLIFPopulation(N, xlim, max_freq, dim=1):
	'''
	GenerateLIFPopulation is deprecated, and is just a wrapper for
	the preferred MakePopulation.
	'''
	return MakePopulation(N, xlim, max_freq, dim)
	

def MakePopulation(N, xlim, max_freq, dim=1):
	'''
	lif = MakePopulation(N, xlim, max_freq, dim=1)

	Generates a random population of LIF neurons.

	Input:
	N is the number of neurons
	xlim is a 2-vector containing the range of inputs [min, max]
	max_freq is a 2-vector with the range of maximim firing frequencies
	dim is the dimensionality of the encoding space (default 1)

	Output:
	lif is a matrix with N columns, where each column holds the
		LIF parameters for a single neuron, in the order:
		0) Vt (threshold voltage)
		1) tau_RC (in seconds)
		2) tau_ref (in seconds)
		3) gain (alpha)
		4) Jbias
		5- encoding vector

	'''
	K = dim # Dimension of vectors being encoded

	lif = zeros([5+K,N])
	xrange = xlim[1] - xlim[0]

	if K==1:
		x_int = random.rand(1,N)*xrange + xlim[0]
	else:
		x_int = random.rand(1,N)*xrange + xlim[0]
		x_int = x_int / 2
		x_theta = random.rand(N) * 360 # Random preferred directions
		x_theta = sort(x_theta)

	for m in range(N):
		tref = 2
		tRC = 20
		eps = tRC / tref
		
		if K==1:
			# enc = +/- 1
			r = random.rand()
			#pdb.set_trace()  # this sets a breakpoint
			enc = sign( (r*xrange)-(x_int[0,m]-xlim[0]) )
			R = enc
			S0 = x_int[0,m]
			idx = array( (enc+1)/2, dtype=int )
			S1 = xlim[idx]  # -1 => xlim[0], 1 => xlim[1]
		elif K==2:
			enc = array([ cos(deg2rad(x_theta[m])), sin(deg2rad(x_theta[m])) ])
			R = 1
			#pdb.set_trace()  # this sets a breakpoints
			S0 = x_int[0,m]
			S1 = xlim[1]
		else:
			enc = random.randn(K)
			R = 1
			S0 = x_int[0,m]
			S1 = xlim[1]
		
		a0 = 1
		a1 = random.rand() * (max_freq[1]-max_freq[0]) + max_freq[0]
		r1 = 1000/(tref*a0)
		r2 = 1000/(tref*a1)
		f1 = (r1-1)/eps
		f2 = (r2-1)/eps
		#pdb.set_trace()  # this sets a breakpoint
		gain = abs( (1.0/(exp(f2)-1) - 1.0/(exp(f1)-1))/(S1-S0) )
		Sthresh = S0 - 1/(R*gain*exp(f1)-1)
		Vt = 1
		Jbias = Vt - R*gain*Sthresh

		lif[0,m] = Vt
		lif[1,m] = 0.001*tRC  # membrane time constant
		lif[2,m] = 0.001*tref  # refractory period
		lif[3,m] = gain
		lif[4,m] = Jbias
		lif[5:,m] = enc	    
	return lif


def Current2Spikes(J, dt, lif=array([[1.,0.02, 0.002, 1, 0, -1]]).T, interp=False):
	'''
    spike_times, V = Current2Spikes(J, dt, lif, interp=False)
    
    Given the input current, outputs the spiking activity of a
    population neurons.
    
    Input:
      J is a PxN array of input currents, with each row containing the
        input current for N neurons at a given time
      dt is the time step (in seconds)
      lif is a (5+K)xN matrix of LIF parameters
      interp determines whether spike times should be interpolated
          (False is the default)
          
    Output:
      spike_times is an array of time-stamps indicating when the neurons
        fired.  In particular,
      V is a PxN array of membrane potentials
    '''
	P = shape(J)[0] # number of x-values
	N = shape(lif)[1] # number of neurons

	T = dt * P # seconds
	t = array(range(P)) * dt  # time steps

	J_th = lif[0,:]
	tau_RC = lif[1,:]
	tau_ref = lif[2,:]
	#alpha = lif[3,:]
	#Jbias = lif[4,:]
	#enc = lif[5,:]

	V = array( zeros(N) )
	Vrec = array( zeros([P,N]) )
	Vrec[0,:] = V

	max_number_spikes = int(floor(T/min(tau_ref)))
	spike_times_matrix = zeros([max_number_spikes,N])
	spike_count = zeros(N, dtype=int)
	refracting = zeros(N)


	for k in range(1,P):
		for m in range(N):
			J_M = J[k-1,m]
			dV = (-1.0 / tau_RC[m]) * (V[m]-J_M * J_th[m])

			#active = (t[k] >= refracting)
			Vn = 0  # default if still in refraction
			if t[k]>=refracting[m]:
				Vo = V[m] # previous V
				if abs(t[k]-refracting[m])>=dt:
					Vn = Vo + dV * dt # new V
				else:
					Vn = Vo + dV * (t[k] - refracting[m])
		
				#Vn = clip(Vn,0,1.e20)
				
				if Vn>=1.0:
					if interp:
						# Linear interpolation of time for threshold crossing
						tstar = ( t[k-1]*(Vn-1) - t[k]*(Vo-1) ) / (Vn-Vo)
					else:
						# Choose t[k] as spike time
						tstar = t[k]
					
					spike_times_matrix[spike_count[m],m] = tstar
					spike_count[m] = spike_count[m] + 1
					refracting[m] = tstar + tau_ref[m]              
					Vn = 0.
				# end of if
				
			# end of refraction if
			V[m] = Vn
			
		# next m
		
		#pdb.set_trace()  # this sets a breakpoint
		Vrec[k,:] = V
		
	# next k
		
	spike_times = []
	for n in range(N):
		spike_times.append(spike_times_matrix[0:spike_count[n],n])

	return spike_times, Vrec


def Stim2Spikes(x, dt, lif, interp=False):
	'''
    spike_times, V = Stim2Spikes(x, dt, LIFparams, interp=False)
    
    Given the input current, outputs the spiking activity of a
    population neurons.
    
    Input:
      J is a PxN array of input currents, with each row containing the
        input current for N neurons at a given time
      dt is the time step (in seconds)
      lif is a (5+K)xN matrix of LIF parameters
      interp determines whether spike times should be interpolated
          (False is the default)
          
    Output:
      spike_times is an array of time-stamps indicating when the neurons
        fired.  In particular,
      V is a PxN array of membrane potentials
    '''

	P = shape(x)[0] # number of x-values
	N = shape(lif)[1] # number of neurons

	T = dt * P # seconds
	t = array(range(P+1)) * dt  # time steps

	#J_th = lif[0,:]
	#tau_RC = lif[1,:]
	#tau_ref = lif[2,:]
	alpha = lif[3,:]
	Jbias = lif[4,:]
	enc = lif[5,:]

	V = array( zeros(N) )
	Vrec = array( zeros([P+1,N]) )
	Vrec[0,:] = V

	J = array( zeros([P,N]) ) # to store input currents
	#pdb.set_trace()  # this sets a breakpoint

	for k in range(1,P):
		for m in range(N):
			J[k,m] = alpha[m]*x[k-1]*enc[m] + Jbias[m]
		# next m
	# next k

	# Pass the work onto the integrator
	spike_times, Vrec = Current2Spikes(J, dt, lif, interp)

	return spike_times, Vrec


def Spikes2PSC(t, spike_times, tau_s, n=0):
	''' psc = Spikes2PSC(t, spike_times, tau_s, n=0) '''
	psc = array(zeros(shape(t)))
	for spike in spike_times:
		psc = psc + PSCFilter(t, tau_s, spike, n)
	return psc


def PSCFilter(t, tau_s, t0=0, n=1):
	''' psc_filter = PSCFilter(t, tau_s, t0=0, n=1)
		t is an array of times,
		tau_s is the synaptic time constant (eg. 0.01 s),
		t0 is the time for the spike,
		n is the power of the curve (see notes).
	'''
	t2 = t-t0
	xc = t2**n * exp(-t2/tau_s)
	xc[t2<0] = 0
	# Normalize the filter so that its integral is 1
	#pdb.set_trace() # this sets a breakpoint
	integral = math.factorial(n)*tau_s**(n+1)
	xc = xc / integral
	return xc


def Encoders(lif):
	'''
	E = Encoders(lif)

	Returns the encoders for the LIF population.

	Input:
	lif is a (5+K)xN matrix containing the LIF parameters
		where K is the dimension of the encoding space
		
	Output:
	E is a KxN matrix containing the encoders, where each columns stores a
		neuron's preferred direction vector (scaled by its gain)
	'''
	E = asmatrix( lif[3,:] * lif[5:,:] ) # encoders
	return E

	
def Biases(lif):
	'''
	beta = Biases(pop)
	
	Returns the biases for all the neurons in the population.
	
	Input:
	  pop is a matrix containing the LIF parameters
	  
	Output:
	  beta is an N vector containing the biases for the neurons.
	'''
	return  lif[4,:]


def Decoders(A, x):
	'''
	D = Decoders(A, x)
	
	Computes the decoding weights for a population of neurons that exhibit
	activity A for input x.

	Input:
	  A is a PxN array, where N is the number of neurons, and P is the
		number of samples of x.
	  x is a PxK array, where each row holds a sample input stimulus from
		a K-dimensional space.

	Output:
	  D is an NxK array, so that A*D ~= x
	'''

	maxval = absolute(A).max()
	A = A + random.randn(shape(A)[0], shape(A)[1])*0.05*maxval # add noise

	xdim = shape(x)
	if len(xdim)==1:
		x = reshape(x,[xdim[0],1])

	#pdb.set_trace()  # this sets a breakpoint
	#U, S, V = linalg.svd(A, full_matrices=False)
	#U,S,V = linalg.svd(A, full_matrices=False)
	#dims = sum(S/S[0]>1e-5) # how many dimensions to keep
	#U = U[:,:(dims-1)]
	#V = V[:(dims-1),:]
	#S = S[:(dims-1)]
	#D = asarray( dot(V.T * asmatrix(diag(1/S)), dot(U.T, x)) )

	D = linalg.lstsq(A,x, rcond=None)[0]

	return D


def Dynamics(x, dt, lif, w, tau_s, tau_m, V0=False):
	'''
	V = Dynamics(x, dt, lif, w, tau_s, tau_m, V0=False)

	Integrate the behaviour of a recurrent network using rate-based LIF nodes.

	Input:
	  x is a PxK array of inputs
	  dt is the time step in seconds (try 0.001)
	  lif is an array holding the LIF parameters in N columns
	  w is an NxN array of recurrent connection weights
	  tau_s is the synaptic time constant (eg. 0.025). If tau_s = 0, then
		 PSPs are assumed to be instantaneous.
	  tau_m is the firing-rate time constant (eg. 0.02). If tau_m = 0, then
		 firing-rate update is instantaneous.
	  V0 (optional) is a 1xN vector of initial firing rate of the nodes
	     (default is zeros)
  
	Output:
	  V is a PxN array holding the activities of the N nodes over
		the P time steps.
    '''
	N = shape(lif)[1]
	P = shape(x)[0] # number of x-values

	if len(shape(x))==1:
		x = x[:,newaxis]
	K = shape(x)[1]  # dimension of input

	T = dt * P
	t = asarray(range(P))*dt

	# Copy out LIF parameters
	J_th = lif[0,:]
	tau_RC = lif[1,:]
	tau_ref = lif[2,:]
	alpha = lif[3,:]
	Jbias = lif[4,:]
	enc = lif[5:,:]

	if type(V0) in (ndarray,list,matrix):
		V = V0
	else:
		V = array( zeros([1,N]) )
	J = dot(V,w) + Jbias
	Vrec = array( zeros([P,N]) )
	Vrec[0,:] = V

	for k in range(1,P):
		#pdb.set_trace()  # this sets a breakpoint
		stim = alpha * asarray( dot(x[k-1,:], enc) )

		if tau_s==0:
			J = dot(V,w) + Jbias + stim
		else:
			dJ = ( dot(V,w) + Jbias + tau_s*stim - J ) / tau_s
			J = J + dJ * dt
	
		# Compute firing rate of each neuron
		A = Current2Rate(J, lif)
	
		if tau_m==0:
			V = A
		else:
			dV = ( 1 / tau_RC ) * (A - V)
			V = V + dV * dt
	
		for m in range(N):
			V[0,m] = max(V[0,m], 0)
		Vrec[k,:] = V
	
	return Vrec
    

def Stim2Rate_ST(x, lif):
	'''
	a = Stim2Rate_ST(x, lif)
	'''
    
	P = shape(x)[0]
	N = shape(lif)[1]
	K = shape(lif)[0] - 5

	A = zeros([P,N])

	alpha = lif[3,:]
	Jbias = lif[4,:]
	enc = lif[5:,:]

	for p in range(K,P):
		alpha = lif[3,:]
		Jbias = lif[4,:]
		enc = lif[5:,:]
		#pdb.set_trace()
		J = asarray(dot(transpose(x[p-K:p]),enc))*alpha + Jbias
		A[p,:] = Current2Rate(J, lif)
		
	return A


def Stim2Rate(x, lif):
	'''
	A = Stim2Rate(J, lif)

	Computes the steady-state firing rate from a stimulus x.

	Input:
	  x is a PxK array of P stimuli, with each of the P rows
		   containing a K-dimensional stimulus.
	  lif is a (5+K)xN matrix of LIF parameters
		  
	Output:
	  A is a PxN array where each row holds the firing rates for N
		neurons for one stimulus value
	'''


	#     N = shape(lif)[1]
	#     x_dims = shape(x)
	#     if len(x_dims)==1:
	#         x_dims = append(x_dims, 1)
	#     P = x_dims[0]  # number of time steps
	#     K = x_dims[1]  # dimension of input
    
	# Copy the relevant LIF parameters
	#J_th = lif[0,:]
	#tau_RC = lif[1,:]
	#tau_ref = lif[2,:]
	alpha = lif[3,:]
	Jbias = lif[4,:]
	enc = lif[5:,:]
	
	#pdb.set_trace()  # this sets a breakpoint
	#pdb.set_trace()
	xdim = shape(x)
	if len(xdim)==1:
		x = reshape(x,[xdim[0],1])
	
	# Convert stimulus to input current
	J = dot(x,enc)*alpha + Jbias
	# And then get Current2Rate to do all the heavy lifting
	A = Current2Rate(J, lif)
	
	return A


def Current2Rate(J, lif):
	'''
	A = Current2Rate(J, lif)

	Computes the steady-state firing rate for input current J.

	Input:
	  J is a PxN array of sets of input currents, with each of the P rows
		   containing N input currents, one for each neuron.
	  lif is a (5+K)xN matrix of LIF parameters
	  
	Output:
	  A is a PxN array where each row holds the firing rates for N
		neurons for one set of input currents
	'''

	# Make sure the input J is a 2-dimensional array.
	if ndim(J)==1:
		print('First input must be a 2-D array')
		return
		
	P, N = shape(J)

	# Copy the useful LIF parameters
	J_th = lif[0,:]
	tau_RC = lif[1,:]
	tau_ref = lif[2,:]

	A = zeros([P,N])

	# Loop over the P different sets of input currents.
	for p in range(P):
		# Compute the theoretical firing rate for each neuron, given
		# its input current.
		for m in range(N):
			if J[p,m]>J_th[m]:
				A[p,m] = 1 / ( tau_ref[m] - tau_RC[m]*log(1-J_th[m]/J[p,m]) )

	return A
    
    
def GenSignal(T, dt, rms, bandwidth, randomSeed):
	'''
	x, A = GenSignal(T, dt, rms, bandwidth, randomSeed)

	Generate a random signal that is bandlimited.

	Input:
	  T is the total time
	  dt is the time step
	  rms is the root mean square
	  bandwidth is [low high], in Hz
	  randomseed is any integer

	Output:
	  x is the signal of length T/dt + 1
	  A holds the Fourier coefficients of the signal.
	'''

	random.seed(randomSeed)

	P = int( floor(T/dt) )   # number of time steps
	t = linspace(0, T, P)

	# MaxFreq = 2 pi / dts (Hz)
	delta_omega = 1.0 / T
	omega = array( range(0,P) )
	h = omega[int(ceil(float(P)/2))]
	omega = delta_omega * (mod(omega+h,P) - h)

	A = array( zeros([P,1]), complex)

	for k in range(P):
		if abs(omega[k])>=bandwidth[0] and abs(omega[k])<=bandwidth[1]:
			A[k] = random.randn() + 1j*random.randn()
	A[0] = random.randn()

	#pdb.set_trace() # this sets a breakpoint

	if (P/2==floor(P/2)):
	   P2 = P/2 - 1
	   for k in range(1,int(P2)):
            A[k] = conj(A[P-k])
	else:
	   P2 = ceil(P/2) - 1
	   for k in range(1,int(P2)):
            A[k] = conj(A[P-k])

	sample_rms = 0.0
	for k in range(P):
		sample_rms = sample_rms + A[k]*conj(A[k])
	sample_rms = sqrt( sample_rms / T * dt )
	A = A / sample_rms * rms * sqrt(P)

	#pdb.set_trace() # this sets a breakpoint
	x = fft.ifft(A,shape(A)[0],0)
	x = real(x)

	return x, A

	

def PlotSpikeRaster(st, y_range=[0, 1.]):
	'''
    PlotSpikeRaster(spiketimes, y_range=[0, 1.])

    Plots a spike raster plot for a list of arrays of spike times.

    Input:
      spiketimes is a list of arrays of spike times, like that returned
          by the function Stim2Spikes.
      y_range is a 2-tuple that holds the y-values that the raster ticks
          should be drawn between
	'''
	N = len(st)  # number of neurons
	
	levels = linspace(y_range[0], y_range[1], N+1, endpoint=True)
	for n in range(N):
		nspikes = len(st[n])
		y = [ [levels[n]]*nspikes , [levels[n+1]]*nspikes ]
		#y = y_range[0] + [levels[n]]*nspikes
		plt.vlines(st[n], levels[n], levels[n+1], color=random.rand(3))
		#plt.plot(vstack((st[n],st[n])), y, color=random.rand(3))
	plt.ylim(y_range)
	return




def CountSpikes(st, tstart, tend):
	'''
	counts = CountSpikes(st, tstart, tend)
	
	Counts how many spikes occur between the start and end times.
	
	Input:
	  st is a list of arrays of spike times, like that returned
	      by the function Stim2Spikes. That is, st[0] is an array of spike
	      times for the first neuron, st[1] is for the next neuron, etc.
	
	Output:
	  counts is an array of integers indicating how many spikes each
	      neuron had.
	'''
	N = len(st)
	r = zeros(N)
	for n in range(N):
		for s in st[n]:
			if (tstart<=s and s<=tend):
				r[n] += 1
	return r

