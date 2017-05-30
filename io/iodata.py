## Input File class to represent arbitrary inputs in a programatic way.
#  Components of the input file are built from InputBlocks.  Enables simple
#  implementation of parametric input files.

from collections import OrderedDict

class InputFile(object):
    ### Set name of input file and associated blocks
    def __init__(self, fname, blocks):

        # Input file name
        self.fname  = fname

        # Store blocks in object
        self.props =[]
        for block in blocks:
            setattr(self, block.name, block)
            self.props.append(block.name)

       #self.blocks_dict = OrderedDict()
       #self.blocks_list = blocks
       #for block in self.blocks_list:
       #    self.blocks_dict.update({block.name:block})

    ### Write Input File
    def write(self):
        InputBlock.writeListOfBlocks(self.fname,
                           [getattr(self,prop) for prop in self.props],self.sep)
        #InputBlock.writeListOfBlocks(self.fname,self.blocks_list,self.sep)

## Input Blcok class to represent arbitrary blocks of input file.  Simply a
#  well-defined class with variable properties.

class InputBlock(object):
    ### Set name of black and property names
    def __init__(self, _name, *args):

        self.name = _name
        self.props = []
        for arg in args:
            if type(arg) in [list, tuple]:
                self.props.append(arg[0])
                if len(arg) == 1:
                    setattr(self, arg[0], [])
                elif len(arg) == 2:
                    setattr(self, arg[0], arg[1])
                else:
                    setattr(self, arg[0], arg[1:])
            elif type(arg) is str:
                setattr(self, arg, []) 
                self.props.append(arg)

    def setProperty(self, prop, val):
        setattr(self, prop, val)

    ## Recursively looks through object and returns a list containing all
    #  InputBlock instances found.  Organized from highest to lowest level in
    #  hierarchy, i.e. out[i+1] cannot depend on out[i]; out[i] MAY depend on
    #  out[i+1]
    #
    def extractListOfInputBlocks(self, args=[]):

        args.insert(0,self)
        for key in self.props[::-1]:
            val = getattr(self,key)
            if 'InputBlock' in val.__class__.__name__:
                val.extractListOfInputBlocks(args)
        return(args)

    def writeToPythonFile(self,fname,mode):

        f = open(fname,mode)
        nspace=15

        # Extract all input blocks that need to be written (reverse the order)
        # to write lower most block first.
        inBlocks=self.extractListOfInputBlocks([])

        # Function for converting block name to variable name
        import re
        def blockToVar(s):

            nameSplit  = [a for a in re.split(r'([A-Z][a-z]*)',s) if a]
            nameMod    = [b[0:4] for b in nameSplit]
            nameMod[0] = nameMod[0].lower()
            name       = ''.join(nameMod)
            return(name)

        f.write('# '+inBlocks[-1].name+'\n')
        for blk in inBlocks:
            f.write(blockToVar(blk.name)+"=AerofInputBlock('"+blk.name+"',\n")
            for p, prop in enumerate(blk.props):
                f.write(' '*nspace+"['"+prop+"',")
                val = getattr( blk , prop )

                if 'InputBlock' in val.__class__.__name__:
                    f.write( blockToVar(val.name) )
                elif val.__class__.__name__ == 'str':
                    f.write( "'"+str( val )+"'" )
                else:
                    f.write( str ( val ) )

                if p < len(blk.props)-1:
                    f.write("],\n")
                else:
                    f.write("]")
            f.write(")\n\n")
        f.close()

    ## Converts a dictionary containing name-value pairs into an instance
    #  of InputBlock.  If a value is a dict, recursively converts into an
    #  InputBlock from the name-value pairs.
    #
    @staticmethod
    def convertDictToBlock(name,dictIn):

        args=[]
        for key in dictIn:
            if (dictIn[key].__class__.__name__ == 'dict' or
                dictIn[key].__class__.__name__ == 'OrderedDict'):
                val = InputBlock.convertDictToBlock(key,dictIn[key])
            else:
                val = dictIn[key]
            args.append([key,val])
        block = InputBlock(name,*args)
        return(block)

    ## Write list of blocks to file
    #
    @staticmethod
    def writeListOfBlocks(fname,listOfBlocks,sep=None):

        open(fname,'w').close() # clear contents of file
        for k, blk in enumerate(listOfBlocks):
            blk.write(fname, 0)
            if ((sep is not None) and (k < len(listOfBlocks)-1)):
                sep(fname)
                #sep(fname)

    ### Takes a list of InputBlocks and combines those with the same name.
    ##  Blocks later in the list have precedence, i.e. two blocks with same
    ##  name and same property will be transformed into a single block whose
    ##  property is that of the block coming later in the list listOfBlocks.
    @staticmethod
    def combineBlocks(listOfBlocks):
        pass
