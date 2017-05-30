import copy
import numpy as np

def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def count_lines(fn):
    return sum(1 for line in open(fn))

def split_line_robust(line):
    """ Robustly split line that may be dlimited by '' and \t """

    line_split0 = [x.rstrip('\n') for x in line.split(' ') if x]
    line_split1 = [x.split('\t') for x in line_split0 if x]
    line_split  = []
    for l_one in line_split1:
        for l_two in l_one:
            if l_two: line_split.append(l_two)
    return(line_split)

def min_bound_box_aligned(nodeset):
    """
    Minimum bounding box aligned with coordinate axes
    """
    from numpy import array
    xmin, xmax, ymin, ymax, zmin, zmax = extents(nodeset)
    nodes = array([ [xmin,ymin,zmin],
                    [xmax,ymin,zmin],
                    [xmin,ymax,zmin],
                    [xmax,ymax,zmin],
                    [xmin,ymin,zmax],
                    [xmax,ymin,zmax],
                    [xmin,ymax,zmax],
                    [xmax,ymax,zmax] ] )
    scale  = ( xmax-xmin , ymax - ymin , zmax - zmin )
    center = ( 0.5*(xmax+xmin) , 0.5*(ymax+ymin) , 0.5*(zmax+zmin) )
    return ( nodes , center , scale )

def extents(nodes):
    """
    Determine extents of 3D mesh
    """
    from numpy import min, max
    return ( min(nodes[:,0]), max(nodes[:,0]),
             min(nodes[:,1]), max(nodes[:,1]),
             min(nodes[:,2]), max(nodes[:,2]) )

def fd_1deriv(f, eps, x, dx, scheme):
    """
    Compute finite difference approximation to first-derivative of f
    """
    if scheme == 'fd1'  :
        f0 = f(x)
        fp = f(x+eps*dx)
        return (1.0/eps)*(fp-f0)
    elif scheme == 'bd1':
        f0 = f(x)
        fm = f(x-eps*dx)
        return (1.0/eps)*(f0-fm)
    elif scheme == 'cd2':
        fp = f(x+eps*dx)
        fm = f(x-eps*dx)
        return (0.5/eps)*(fp-fm)
    elif scheme == 'cd6':
        fp3 = f(x+3*eps*dx)
        fp2 = f(x+2*eps*dx)
        fp1 = f(x+eps*dx)
        fm1 = f(x-eps*dx)
        fm2 = f(x-2*eps*dx)
        fm3 = f(x-3*eps*dx)
        return (1.0/60.0/eps)*(45.0*(fp1-fm1)-9.0*(fp2-fm2)+(fp3-fm3))

class History(object):
    def __init__(self):
        self.count = -1
        self.param = []
        self.val   = None

    def __call__(self):
        return self.count

    def check_parameter(self, param):

        from numpy import all, ndarray

        in_hist = False
        for p in self.param:
            # Handle dictionaries (AEROF parameters) and numpy arrays (SDESIGN/
            # BLENDER parameters) differently.
            if type(param) is dict:
                all_true = True
                for key in param:
                    if not all(p[key] == param[key]):
                        all_true = False
                        break

                in_hist = all_true
                if in_hist: break

            if type(param) is ndarray:
                if all(p == param):
                   in_hist = True
                   break
        return ( in_hist )

    def increment(self, param, check=False, val=None):

        if not check or not self.check_parameter(param):
            self.param.append(copy.deepcopy(param))
            self.count += 1
            self.val = val

class Object(object):
    """
    Object class for storing various properties using '.' syntax instead of
    dict syntax.
    """
    def __init__(self,id,type,**kwargs):
        self.id   = id
        self.type = type
        self.addProperties(**kwargs)

    def addProperties(self,**kwargs):
        for s in kwargs:
            setattr(self,s,kwargs[s])

class Group(object):
    """
    Group class to turn a list of items into a group where id-based operations
    available (i.e. search by id property).
    """
    def __init__(self, list_of_items):
        self.items = list_of_items
        self.n     = len(list_of_items)

    def get_ids(self):
        """
        Return a list containing the ids of all items in the group (in the order
        they appear).
        """
        return [item.id for item in self.items]

    def add_to_group(self,item):
        """
        Add item to the group and increment the counter.
        """
        self.items.append(item)
        self.n += 1

    def get_from_id(self,id=None):
        """
        Get an item from the group based on its id. If id = None, returns all
        items in the group; if id is an int, returns the first item in the list
        with a matching 'id' property; if id is a list, returns all items in
        the group that have an id contained in the list.
        """
        if id is None:
            return(self.items)
        if type(id) is int:
            for item in self.items:
                if item.id == id: return(item)
        items=[]
        if type(id) is list:
            return ([item.id for item in self.items if (item.id in id)])

class Database(object):
    def __init__(self, norm=np.linalg.norm, offset=0):
        self.mu_db = []
        self.which = []
        self.norm  = norm
        self.offset = offset

    def find(self, mu):
        """ Find entry in database """
        for k, muk in enumerate(self.mu_db):
            if self.norm(muk - mu) == 0.0:
                ind = k+self.offset
                return ind, self.which[k]
        return None, None

    #def check(self, mu):
    #    """ Check if entry exists in database """
    #    for k, muk in enumerate(self.mu_db):
    #        if self.norm(muk - mu) == 0.0:
    #            ind = k+self.offset
    #            return ind, 'prim' in self.which[k], 'dual' in self.which[k]
    #    return None, False, False

    def check(self, mu):
        """ Check if entry in database exits """
        return self.find(mu)[0] is not None

    def find_which(self, mu, which_to_find):
       """ Find 'which' in entry of database """
       ind, which = self.find(mu)
       if ind is not None and which_to_find in which:
           return which[which_to_find]
       else:
           return None

    def check_which(self, mu, which_to_find):
       """ Check entry of database for 'which' """
       return self.find_which(mu, which_to_find) is not None

    def add(self, mu, **which):
        """ Add parameter to database """
        # Handle case where mu already seen
        k, _ = self.find(mu)
        if k is not None:
            self.which[k].update(which)
            return k
        # Handle case where mu not seen
        self.mu_db.append(mu.copy())
        self.which.append(which)
        return self.offset+len(self.mu_db)-1

    def current(self):
        return self.offset+len(self.mu_db)-1

    def reset(self):
        return Database(norm=self.norm, offset=self.offset)

class Database0(object):
    def __init__(self, norm=np.linalg.norm, offset=0):
        self.mu_db = []
        self.which = []
        self.norm  = norm
        self.offset = offset

    def check(self, mu):
        """ Check if entry exists in database """
        for k, muk in enumerate(self.mu_db):
            if self.norm(muk - mu) == 0.0:
                ind = k+self.offset
                return ind, 'prim' in self.which[k], 'dual' in self.which[k]
        return None, False, False

    def find(self, mu):
        """ Find entry in database """
        for k, muk in enumerate(self.mu_db):
            if self.norm(muk - mu) == 0.0:
                return k+self.offset
        return None

    def add(self, mu, which):
        """ Add parameter to database """
        # Handle case where mu already seen
        k = self.find(mu)
        if k is not None:
            self.which[k].add(which)
            return k
        # Handle case where mu not seen
        self.mu_db.append(mu.copy())
        self.which.append(set([which]))
        return self.offset+len(self.mu_db)-1

    def current(self):
        return self.offset+len(self.mu_db)-1

    def reset(self):
        return Database(norm=self.norm, offset=self.offset)
