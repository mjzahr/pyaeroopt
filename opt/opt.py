## @package opt
#  Optimization Interface
#
#  Class for interfacing to various python optimization solvers (pyOpt,
#  scipy.optimize, nlopt currently supported)

import numpy as np
import scipy.optimize
import pyOpt
#import openopt
#import nlopt

class Optimize(object):
    ## Initialize optimization class with empty/default properties.  Objective
    #  function and constraints can be added via calls to other methods
    #  addObjective, ...
    def __init__(self,**kwargs):

        # Initialize relevant counters, opt variable itertes, \pm \infty
        self.counters={}
        self.varIter = []
        self.inf  =  1.0e20 if 'inf'  not in kwargs else kwargs['inf']
        self.minf = -1.0e20 if 'minf' not in kwargs else kwargs['minf']

        # Initialize variable-related properties
        self.nvar      = 0
        self.varInit   = np.zeros(self.nvar,dtype='float')
        self.varBounds = np.zeros((self.nvar,2),dtype='float')

        # Initialize objective-related properties
        self.objective = None
        self.gradient  = None

        # Initialize constraint-related properties
        self.nLinConstr = 0 
        self.linMat     = np.zeros((self.nLinConstr,self.nvar),dtype='float')
        self.linBounds  = np.zeros((self.nLinConstr,2),dtype='float')

        self.nNonlinConstr = 0
        self.nonlinConstr  = []
        self.nonlinJac     = []
        self.nonlinBounds  = np.zeros((self.nNonlinConstr,2),dtype='float')
        self.nNonlinConstrPerGroup = np.zeros(1,dtype='int')

    def addVariables(self,nvar,init,low,up):

        self.nvar += nvar
        self.varInit   = np.r_[self.varInit,init]
        self.varBounds = np.r_[self.varBounds,np.c_[low,up]]
        # Resize linear matrix
        self.linMat.resize((self.nLinConstr,self.nvar))

    def addObjective(self,obj,grad=None,hess=None,hessVec=None):

        self.objective    = obj
        self.gradient     = grad
        self.hessian      = hess
        self.hessianVec   = hessVec

    def addLinConstraints(self,nconstr,mat,low,up):

        self.nLinConstr += nconstr
        self.linMat    = np.r_[self.linMat,mat]
        self.linBounds = np.r_[self.linBounds,np.c_[low,up]]

    def addNonlinConstraints(self,nconstr,constr,jac,low,up):

        self.nNonlinConstr += nconstr
        self.nonlinConstr.append(constr)
        self.nonlinJac.append(jac)
        self.nonlinBounds = np.r_[self.nonlinBounds,np.c_[low,up]]
        self.nNonlinConstrPerGroup = np.r_[self.nNonlinConstrPerGroup,
                                           np.array([nconstr])]

    def evalNonlinConstraints(self, param, which, ind=None):

        # Determine constraints to evaluate
        if ind is None: ind = np.arange(self.nNonlinConstr)
        if type(ind) is int: ind = np.array([ind])

        # Initialize arrays for constraints and/or jacobians
        if which in ['constr','both']:
            constrs = np.zeros(len(ind),dtype=float)
        if which in ['jac','both']:
            jacs = np.zeros((len(ind),self.nvar),dtype=float)

        ind_per_group = []
        # Determine indices per group
        for g in range(len(self.nonlinConstr)):
            lb = sum(self.nNonlinConstrPerGroup[:g+1])
            ub = sum(self.nNonlinConstrPerGroup[:g+2])
            ind_below = ind[ np.logical_and(ind < ub, ind >= lb) ]
            ind_per_group.append( ind_below )

        # Evaluate appropriate constraints and/or jacobians
        cnt = 0
        for g in range(len(self.nonlinConstr)):
            if len(ind_per_group[g]) > 0:
                idx = ind_per_group[g]
                if which in ['constr','both']:
                    constrs[cnt:cnt+len(idx)] = self.nonlinConstr[g](param)[idx]
                if which in ['jac','both']:
                    jacs[cnt:cnt+len(idx)]    = self.nonlinJac[g](param)[idx, :]
                cnt += len(idx)
        #for cnt, i in enumerate(ind):
        #    group=(self.nNonlinConstrPerGroup.cumsum() < i+1).nonzero()[0][-1]
        #    index=i-self.nNonlinConstrPerGroup[:group+1].sum() 
        #    if which in ['constr','both']:
        #        constrs[cnt] = self.nonlinConstr[group](param, index)
        #    if which in ['jac','both']:
        #        jacs[cnt]    = self.nonlinJac[group](param,index)

        if which is 'constr': return (constrs)
        if which is 'jac'   : return (jacs)
        if which is 'both'  : return (constrs,jacs)

    def optimize(self,solver,sens='finite-diff',options=None,callback=None):

        if 'pyopt' in solver:
            xStar,fStar = self.optimizePyopt(solver,sens,options)
        elif 'scipy' in solver:
            xStar,fStar = self.optimizeScipy(solver,sens,options,callback)
        elif 'nlopt' in solver:
            xStar,fStar = self.optimizeNlopt(solver,sens,options)
        elif 'openopt' in solver:
            xStar,fStar = self.optimizeOpenopt(solver,sens,options)

        return ( xStar, fStar )

    #TODO: test
    def optimizePyopt(self,solver,sens,options):

        # pyOpt requires empty options to be specified as {}, not None
        if options is None: options = {}

        eqConstr = ( 'SLSQP'    not in solver and
                     'CONMIN'   not in solver and
                     'COBYLA'   not in solver and
                     'FILTERSD' not in solver and
                     'SDPEN'    not in solver )

        # Organize constraints
        indLinEq = []; indLinIneq = []; indNonlinEq = []; indNonlinIneq = [];
        for k, bnd in enumerate(self.linBounds):
            if bnd[0]==bnd[1] and eqConstr: indLinEq.append(k)
            else:                             indLinIneq.append(k)
        indLinEq=np.array(indLinEq); indLinIneq=np.array(indLinIneq);
        for k, bnd in enumerate(self.nonlinBounds):
            if bnd[0]==bnd[1] and eqConstr: indNonlinEq.append(k)
            else:                             indNonlinIneq.append(k)
        indNonlinEq=np.array(indNonlinEq);indNonlinIneq=np.array(indNonlinIneq);

        # pyOpt objective
        def objectivePyopt(xIn,*args,**kwargs):

            x = xIn[:self.nvar]

            f = self.objective(x)
            g = np.zeros( 0, dtype = float)
            if len(indLinEq) > 0:
                g = np.r_[ g, np.dot(self.linMat[indLinEq,:],x)]
            if len(indNonlinEq) > 0:
                g =np.r_[g, self.evalNonlinConstraints(x,'constr',indNonlinEq)]
            if len(indLinIneq) > 0:
                g = np.r_[ g, np.dot(self.linMat[indLinIneq,:],x)]
            if len(indNonlinIneq) > 0:
                g=np.r_[g, self.evalNonlinConstraints(x,'constr',indNonlinIneq)]

            fail = 0
            if f >= self.inf or np.any( g >= self.inf): fail = 1

            return f, g, fail

        # pyOpt gradient
        def gradientPyopt(xIn,f,g,*args,**kwargs):

            x = xIn[:self.nvar]

            df = self.gradient(x)
            dg = np.zeros( (0,self.nvar), dtype = float)
            if len(indLinEq) > 0:
                dg = np.r_[ dg, self.linMat[indLinEq,:] ]
            if len(indNonlinEq) > 0:
                dg= np.r_[ dg, self.evalNonlinConstraints(x,'jac',indNonlinEq) ]
            if len(indLinIneq) > 0:
                dg = np.r_[ dg, self.linMat[indLinIneq,:] ]
            if len(indNonlinIneq) > 0:
                dg=np.r_[dg, self.evalNonlinConstraints(x,'jac',indNonlinIneq)]

            fail = 0
            if f >= self.inf or np.any( g >= self.inf): fail = 1

            return df.reshape((1,-1)), dg, fail

        # Instantiate optimization problem
        optProb = pyOpt.Optimization('pyopt',objectivePyopt)

        # Add objective
        optProb.addObj('objective')

        # Add variables
        optProb.addVarGroup('var',self.nvar,type='c',value=self.varInit,
                            lower=self.varBounds[:,0],upper=self.varBounds[:,1])
        # Add constraints
        if len(indLinEq) > 0:
            optProb.addConGroup('lin-equality',len(indLinEq),type='e',
                                               equal=self.linBounds[indLinEq,0])
        if len(indNonlinEq) > 0:
            optProb.addConGroup('nonlin-equality',len(indNonlinEq),type='e',
                                         equal=self.nonlinBounds[indNonlinEq,0])
        if len(indLinIneq) > 0:
            optProb.addConGroup('lin-inequality',len(indLinIneq),type='i',
                                             lower=self.linBounds[indLinIneq,0],
                                             upper=self.linBounds[indLinIneq,1])
        if len(indNonlinIneq) > 0:
            optProb.addConGroup('nonlin-inequality',len(indNonlinIneq),type='i',
                                       lower=self.nonlinBounds[indNonlinIneq,0],
                                       upper=self.nonlinBounds[indNonlinIneq,1])

        # Setup solver
        if 'SNOPT' in solver:
            optimizer = pyOpt.SNOPT(options=options)
        if 'SLSQP' in solver:
            optimizer = pyOpt.SLSQP(options=options)
        if 'CONMIN' in solver:
            optimizer = pyOpt.CONMIN(options=options)
        if 'ALGENCAN' in solver:
            optimizer = pyOpt.ALGENCAN(options=options)
        if 'ALPSO' in solver:
            optimizer = pyOpt.ALPSO(options=options)
        if 'ALHSO' in solver:
            optimizer = pyOpt.ALHSO(options=options)
        if 'COBYLA' in solver:
            optimizer = pyOpt.COBYLA(options=options)
        if 'FILTERSD' in solver:
            optimizer = pyOpt.FILTERSD(options=options)
        if 'KOPT' in solver:
            optimizer = pyOpt.KOPT(options=options)
        if 'MIDACO' in solver:
            optimizer = pyOpt.MIDACO(options=options)
        if 'KSQP' in solver:
            optimizer = pyOpt.KSQP(options=options)
        if 'SDPEN' in solver:
            optimizer = pyOpt.SDPEN(options=options)
        if 'SOLVOPT' in solver:
            optimizer = pyOpt.SOLVOPT(options=options)

        # Run optimization
        if sens == 'finite-difference':
            optimizer(optProb,sens_type='FD')
        else:
            optimizer(optProb,sens_type=gradientPyopt)

        # Extract solution
        j = len(optProb._solutions)-1
        xStar = np.zeros(self.nvar)
        for k in range(self.nvar):
            xStar[k] = optProb._solutions[j].getVar(k).value

        return xStar,objectivePyopt(xStar)[0]

    #TODO: test
    def optimizeNlopt(self,solver,sens,options):

        optProb = nlopt.opt( getattr( nlopt , solver.lstrip('nlopt:') ),
                             self.nvar)

        def objectiveNlopt(x,grad):

            if grad.size > 0:
                grad[:] = self.gradient( x )

            return ( self.objective( x ) )

        # Organize constraints
        eqConstr = ( 'MMA' not in solver and
                     'COBYLA' not in solver )

        indLinEq = []; indLinIneqUp = []; indLinIneqLow = [];
        indNonlinEq = []; indNonlinIneqUp = []; indNonlinIneqLow = [];
        for k, bnd in enumerate(self.linBounds):
            if bnd[0] == bnd[1] and eqConstr: indLinEq.append(k)
            else:
                if bnd[0] > self.minf : indLinIneqLow.append(k)
                if bnd[1] < self.inf  : indLinIneqUp.append(k)
        indLinEq=np.array(indLinEq)
        indLinIneqUp=np.array(indLinIneqUp)
        indLinIneqLow=np.array(indLinIneqLow)

        for k, bnd in enumerate(self.nonlinBounds):
            if bnd[0] == bnd[1] and eqConstr: indNonlinEq.append(k)
            else:
                if bnd[0] > self.minf : indNonlinIneqLow.append(k)
                if bnd[1] < self.inf  : indNonlinIneqUp.append(k)
        indNonlinEq=np.array(indNonlinEq)
        indNonlinIneqUp=np.array(indNonlinIneqUp)
        indNonlinIneqLow=np.array(indNonlinIneqLow)

        # linear constraints
        linEqConstr = np.zeros( (0,self.nvar) , dtype=float )
        linEqRhs    = np.zeros( 0 , dtype=float )
        if len(indLinEq) > 0:
            linEqConstr = self.linMat[indLinEq,:]
            linEqRhs    = self.linBounds[indLinEq,0]

        linIneqConstr = np.zeros( (0,self.nvar) , dtype=float )
        linIneqRhs    = np.zeros( 0 , dtype=float )
        if len(indLinIneqLow) > 0:
            linIneqConstr = np.r_[ linIneqConstr,
                                   -self.linMat[indLinIneqLow,:] ]
            linIneqRhs    = np.r_[ linIneqRhs,
                                   -self.linBounds[indLinIneqLow,0] ]
        if len(indLinIneqUp) > 0:
            linIneqConstr = np.r_[ linIneqConstr,
                                   self.linMat[indLinIneqUp,:] ]
            linIneqRhs    = np.r_[ linIneqRhs,
                                   self.linBounds[indLinIneqUp,1] ]

        # constraints
        def linEqConstrNlopt(results,x,grad):

            results[:] = np.dot(linEqConstr,x) - linEqRhs
            if grad.size > 0:
                grad[:] = linEqConstr

        def linIneqConstrNlopt(results,x,grad):

            results[:] = np.dot(linIneqConstr,x) - linIneqRhs
            if grad.size > 0:
                grad[:] = linIneqConstr

        def nonlinEqConstrNlopt(results,x,grad):

            results[:] = (self.evalNonlinConstraints(x,'constr',indNonlinEq) -
                                               self.nonlinBounds[indNonlinEq,0])
            if grad.size > 0:
                grad[:] = self.evalNonlinConstraints(x,'jac',indNonlinEq)

        def nonlinIneqConstrNlopt(results,x,grad):

            tmp = np.zeros( 0, dtype=float )
            if len(indNonlinIneqLow) > 0:
                tmp = np.r_[ tmp,
                  -self.evalNonlinConstraints(x,'constr',indNonlinIneqLow) +
                                          self.nonlinBounds[indNonlinIneqLow,0]]
            if len(indNonlinIneqUp) > 0:
                tmp = np.r_[ tmp,
                   self.evalNonlinConstraints(x,'constr',indNonlinIneqUp)  -
                                          self.nonlinBounds[indNonlinIneqUp,1]]
            results[:] = tmp

            if grad.size > 0:
                tmpGrad = np.zeros( (0,self.nvar), dtype=float)
                if len(indNonlinIneqLow) > 0:
                    tmpGrad = np.r_[ tmpGrad,
                          -self.evalNonlinConstraints(x,'jac',indNonlinIneqLow)]
                if len(indNonlinIneqUp ) > 0:
                    tmpGrad = np.r_[ tmpGrad,
                           self.evalNonlinConstraints(x,'jac',indNonlinIneqUp)]
                grad[:] = tmpGrad
 
        optProb.set_min_objective( objectiveNlopt )
        optProb.set_lower_bounds(self.varBounds[:,0])
        optProb.set_upper_bounds(self.varBounds[:,1])

        if linEqConstr.shape[0] > 0:
            optProb.add_equality_mconstraint( linEqConstrNlopt ,
                              1.0e-6*np.ones(linEqConstr.shape[0],dtype=float) )
        if indNonlinEq.shape[0] > 0:
            optProb.add_equality_mconstraint( nonlinEqConstrNlopt ,
                              1.0e-6*np.ones(indNonlinEq.shape[0],dtype=float) )
        if linIneqConstr.shape[0] > 0:
            optProb.add_inequality_mconstraint( linIneqConstrNlopt ,
                            1.0e-6*np.ones(linIneqConstr.shape[0],dtype=float) )
        if indNonlinIneqLow.shape[0] + indNonlinIneqUp.shape[0] > 0:
            optProb.add_inequality_mconstraint( nonlinIneqConstrNlopt ,
                                   1.0e-6*np.ones(indNonlinIneqLow.shape[0]+
                                                  indNonlinIneqUp.shape[0],
                                                  dtype=float) )

        optProb.set_xtol_abs(1.0e-8)
        xopt = optProb.optimize(self.varInit)

        return xopt, self.objective( xopt )

    #TODO: test
    def optimizeScipy(self,solver,sens,options,callback=None):

        # Reformat variable bounds
        bnds = [(x[0] if x[0] > self.minf else None,
                 x[1] if x[1] < self.inf  else None) for x in self.varBounds]

        # Need to define alternate functions to handle scoping issues with
        # lambdas (specifically, when trying to use index in a loop to
        # extract specific constraints to put into scipy format).  This will
        # essentially define new scopes.
        def bndValFactory(k,which):
           if which == 'low': 
               return (lambda x: x[k] - self.varBounds[k,0])
           if which == 'up': 
               return (lambda x: self.varBounds[k,1] - x[k])
        def bndGradFactory(k,which):
            ek = np.zeros(self.nvar,dtype=float)
            ek[k]=1.0
            if which == 'low':
                return (lambda x: ek)
            if which == 'up':
                return (lambda x: -ek)
        def linValFactory(k,which):
            if which == 'low':
                return(lambda x: np.dot(self.linMat[k,:],x)-self.linBounds[k,0])
            if which == 'up':
                return(lambda x: self.linBounds[k,1]-np.dot(self.linMat[k,:],x))
        def linGradFactory(k,which):
            if which == 'low':
                return (lambda x: self.linMat[k,:])
            if which == 'up':
                return (lambda x: -self.linMat[k,:])
        def nonlinValFactory(k,which):
            if which == 'low':
                return(lambda x: self.evalNonlinConstraints(x,'constr',k) -
                                                         self.nonlinBounds[k,0])
            if which == 'up':
                return(lambda x: self.nonlinBounds[k,1] -
                                       self.evalNonlinConstraints(x,'constr',k))
        def nonlinGradFactory(k,which):
            if which == 'low':
                return(lambda x: self.evalNonlinConstraints(x,'jac',k))
            if which == 'up':
                return(lambda x: -self.evalNonlinConstraints(x,'jac',k))

        # Reformat constraints (don't distinguish between lin/nonlin)
        constrs = []
        # Add linear equality constraints
        if self.nLinConstr > 0:
            for k, bnd in enumerate(self.linBounds):
                if bnd[0] == bnd[1]:
                    constrs.append({'type':'eq',
                                   'fun': linValFactory(k,'low'),
                                   'jac': linGradFactory(k,'low')})

        # Add nonlinear equality constraints
        if self.nNonlinConstr > 0:
            for k, bnd in enumerate(self.nonlinBounds):
                if bnd[0] == bnd[1]:
                    if None in self.nonlinJac:
                        constrs.append({'type':'eq',
                                   'fun': nonlinValFactory(k,'low')})
                    else:
                        constrs.append({'type':'eq',
                                   'fun': nonlinValFactory(k,'low'),
                                   'jac': nonlinGradFactory(k,'low')})

        # Add linear inequality constraints
        if self.nLinConstr > 0:
            for k, bnd in enumerate(self.linBounds):
                if bnd[0] < bnd[1]:
                    if bnd[0] > self.minf:
                        constrs.append({'type':'ineq',
                                   'fun': linValFactory(k,'low'),
                                   'jac': linGradFactory(k,'low')})
                    if bnd[1] < self.inf:
                        constrs.append({'type':'ineq',
                                   'fun': linValFactory(k,'up'),
                                   'jac': linGradFactory(k,'up')})

        # Add nonlinear inequality constraints
        if self.nNonlinConstr > 0:
            for k, bnd in enumerate(self.nonlinBounds):
                if bnd[0] < bnd[1]:
                    if bnd[0] > self.minf:
                        if None in self.nonlinJac:
                            constrs.append({'type':'ineq',
                                        'fun': nonlinValFactory(k,'low')})
                        else:
                            constrs.append({'type':'ineq',
                                        'fun': nonlinValFactory(k,'low'),
                                        'jac': nonlinGradFactory(k,'low')})
                    if bnd[1] < self.inf:
                        if None in self.nonlinJac:
                            constrs.append({'type':'ineq',
                                        'fun': nonlinValFactory(k,'up')})
                        else:
                            constrs.append({'type':'ineq',
                                        'fun': nonlinValFactory(k,'up'),
                                        'jac': nonlinGradFactory(k,'up')})

        # If using COBYLA, add simple bounds as general constraints and
        # treat all equality constraints as two inequality constraints
        if 'COBYLA' in solver:
            constrs = []

            # Add linear constraints
            if self.nLinConstr > 0:
                for k, bnd in enumerate(self.linBounds):
                    if bnd[0] > self.minf:
                        constrs.append({'type':'ineq',
                                        'fun': linValFactory(k,'low'),
                                        'jac': linGradFactory(k,'low')})
                    if bnd[1] < self.inf:
                        constrs.append({'type':'ineq',
                                        'fun': linValFactory(k,'up'),
                                        'jac': linGradFactory(k,'up')})

            # Add nonlinear constraints
            if self.nNonlinConstr > 0:
                for k, bnd in enumerate(self.nonlinBounds):
                    if bnd[0] > self.minf:
                        constrs.append({'type':'ineq',
                                        'fun': nonlinValFactory(k,'low'),
                                        'jac': nonlinGradFactory(k,'low')})
                    if bnd[1] < self.inf:
                        constrs.append({'type':'ineq',
                                        'fun': nonlinValFactory(k,'up'),
                                        'jac': nonlinGradFactory(k,'up')})

            # Add bound constraints
            for k, bnd in enumerate(self.varBounds):
                ek = np.zeros(self.nvar,dtype=float)
                ek[k]=1.0
                if bnd[0] > self.minf:
                    constrs.append({'type':'ineq',
                                    'fun': bndValFactory(k,'low'),
                                    'jac': bndGradFactory(k,'low')})
                if bnd[1] < self.inf:
                    constrs.append({'type':'ineq',
                                    'fun': bndValFactory(k,'up'),
                                    'jac': bndGradFactory(k,'up')})

        summ = scipy.optimize.minimize(self.objective, self.varInit,
                                method=solver.lstrip('scipy:'),bounds=bnds,
                                jac=self.gradient,constraints=constrs,
                                hess=self.hessian, hessp=self.hessianVec,
                                callback=callback, options=options)
        print (summ)
        return ( summ.x, summ.fun)

    #TODO: test
    def optimizeOpenopt(self,solver,sens,options,callback=None):

        # Organize constraints
        indLinEq = []; indLinIneqUp = []; indLinIneqLow = [];
        indNonlinEq = []; indNonlinIneqUp = []; indNonlinIneqLow = [];
        for k, bnd in enumerate(self.linBounds):
            if bnd[0] == bnd[1]   : indLinEq.append(k)
            if bnd[0] > self.minf : indLinIneqLow.append(k)
            if bnd[1] < self.inf  : indLinIneqUp.append(k)
        indLinEq=np.array(indLinEq)
        indLinIneqUp=np.array(indLinIneqUp)
        indLinIneqLow=np.array(indLinIneqLow)

        for k, bnd in enumerate(self.nonlinBounds):
            if bnd[0] == bnd[1]   : indNonlinEq.append(k)
            if bnd[0] > self.minf : indNonlinIneqLow.append(k)
            if bnd[1] < self.inf  : indNonlinIneqUp.append(k)
        indNonlinEq=np.array(indNonlinEq)
        indNonlinIneqUp=np.array(indNonlinIneqUp)
        indNonlinIneqLow=np.array(indNonlinIneqLow)

        # linear constraints
        linEqConstr = np.zeros( (0,self.nvar), dtype=float )
        linEqRhs    = np.zeros( 0 , dtype=float )
        if len(indLinEq) > 0:
            linEqConstr = self.linMat[indLinEq,:],
            linEqRhs    = self.linBounds[indLinEq,0]

        linIneqConstr = np.zeros( (0,self.nvar), dtype=float )
        linIneqRhs    = np.zeros( 0 , dtype=float )
        if len(indLinIneqLow) > 0:
            linIneqConstr = np.r_[ linIneqConstr,
                                   -self.linMat[indLinIneqLow,:] ]
            linIneqRhs    = np.r_[ linIneqRhs,
                                   -self.linBounds[indLinIneqLow,0] ]
        if len(indLinIneqUp) > 0:
            linIneqConstr = np.r_[ linIneqConstr,
                                   self.linMat[indLinIneqUp,:] ]
            linIneqRhs    = np.r_[ linIneqRhs,
                                   self.linBounds[indLinIneqUp,1] ]

        # nonlinear constraints
        def nonlinEqConstrOpenopt(x):

            if len(indNonlinEq) > 0:
                c = (self.evalNonlinConstraints(x,'constr',indNonlinEq) -
                                               self.nonlinBounds[indNonlinEq,0])
                return ( c )

        def nonlinEqJacOpenopt(x):

            if len(indNonlinEq) > 0:
                j = self.evalNonlinConstraints(x,'jac',indNonlinEq)
                return ( j )

        def nonlinIneqConstrOpenopt(x):

            c = np.zeros(0,dtype=float)
            if len(indNonlinIneqLow) > 0:
                c = np.r_[ c,-self.evalNonlinConstraints(x,'constr',
                       indNonlinIneqLow) +self.nonlinBounds[indNonlinIneqLow,0]]
            if len(indNonlinIneqUp) > 0:
                c = np.r_[ c, self.evalNonlinConstraints(x,'constr',
                        indNonlinIneqUp) - self.nonlinBounds[indNonlinIneqUp,1]]
            return ( c )
 
        def nonlinIneqJacOpenopt(x):

            j = np.zeros((0,self.nvar),dtype=float)
            if len(indNonlinIneqLow) > 0:
                j = np.r_[j,
                    -self.evalNonlinConstraints(x,'jac',indNonlinIneqLow) +
                                          self.nonlinBounds[indNonlinIneqLow,0]]
            if len(indNonlinIneqUp) > 0:
                j = np.r_[j,
                   self.evalNonlinConstraints(x,'jac',indNonlinIneqUp)  -
                                          self.nonlinBounds[indNonlinIneqUp,1]]

            return ( j )

        # Turn unused contraints into None
        if linEqConstr.shape[0] == 0:
            linEqConstr = None
            linEqRhs    = None 
        if linIneqConstr.shape[0] == 0:
            linIneqConstr = None
            linIneqRhs    = None 
        if len(indNonlinEq) == 0:
            h  = None
            dh = None
        if len(indNonlinIneqUp) + len(indNonlinIneqLow) == 0:
            c  = None
            dc = None
 
        optProb = openopt.NLP(self.objective, self.varInit,
                              df=self.gradient,
                              lb=self.varBounds[:,0],
                              ub=self.varBounds[:,1],
                              Aeq=linEqConstr,beq=linEqRhs,
                              A=linIneqConstr,b=linIneqRhs,
                              h=nonlinEqConstrOpenopt,dh=nonlinEqJacOpenopt,
                              c=nonlinIneqConstrOpenopt,dc=nonlinIneqJacOpenopt)

        summ = optProb.solve(solver.lstrip('openopt:'))
        return summ.xf,summ.ff
