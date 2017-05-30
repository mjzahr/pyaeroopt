import os, numbers

from pyaeroopt.interface import CodeInterface
from pyaeroopt.io.iodata import InputBlock, InputFile

class Aerof(CodeInterface):
    """
    An object to facilitate interfacing to AERO-F.
    """
    def __init__(self, **kwargs):

        # Constructor of base class, check fields, and extract anticipated input
        super(Aerof, self).__init__(**kwargs)

        if self.bin is None: self.bin = os.path.expandvars('$AEROF')

    def read_ascii_out(self):
        pass

    def create_input_file(self, p, desc_ext=None, db=None):
        fname, log = desc_ext[0], desc_ext[1]
        self.infile = AerofInputFile(fname, desc_ext[2:], log)

class AerofInputFile(InputFile):
    def __init__(self, fname, blocks, log='aerof.tmp.log', **kwargs):
        super(AerofInputFile, self).__init__(fname, blocks)
        self.sep = AerofInputBlock.line_break
        self.log = log
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

class AerofInputBlock(InputBlock):
    def __init__(self, name, *args):
        super(AerofInputBlock, self).__init__(name, *args)

    def write(self, fname, indent_level):

        # Keywords under Input, Postpro, Restart, .. that do NOT designate paths
        # i.e. do not place in ""
        notpathvars=['ShapeDerivativeType', 'OptimalPressureDimensionality']

        blockvars = vars(self)
        indent = '   '

        f=open(fname,'a+')
        f.write(indent*indent_level+'under '+self.name+' {\n')
        for prop in self.props: #blockvars:
            if prop == 'name': continue
            if blockvars[prop] is None: continue

            if type(blockvars[prop]) == str:
                if (( prop not in notpathvars ) and (self.name in ["Input",
                                              "Postpro","Restart","Directories",
                                                      "Files","NonlinearROM"])):
                    tmp='"'+blockvars[prop]+'"'
                else:
                    tmp=blockvars[prop]
                f.write(indent*(indent_level+1)+prop+' = '+tmp+';')
            if isinstance(blockvars[prop], numbers.Number):
                f.write(indent*(indent_level+1)+prop+' = '+
                                                       str(blockvars[prop])+';')
            if isinstance(blockvars[prop],InputBlock):
                f.close()
                blockvars[prop].write(fname,indent_level+1)
                f=open(fname,'a+')
            f.write('\n')
        f.write(indent*indent_level+'}')
        f.close()

    @staticmethod
    def line_break(fname):
        f=open(fname,'a'); f.write('\n\n'); f.close();

    @staticmethod
    def from_dict(block):
        pass

    @staticmethod
    def from_input_block(block):
        pass
