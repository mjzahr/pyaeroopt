import numbers

from pyaeroopt.interface import CodeInterface
from pyaeroopt.io        import InputBlock, InputFile

class Aeros(CodeInterface):
    """
    An object to facilitate interfacing to AERO-S.
    """
    def __init__(self, **kwargs):

        # Constructor of base class, check fields, and extract anticipated input
        super(Aeros, self).__init__(**kwargs)

        if self.bin is None: self.bin = os.path.expandvars('$AEROS')

    def read_ascii_out(self):
        pass

class AerosInputFile(InputFile):
    def __init__(self, fname, blocks, log='aeros.tmp.log', **kwargs):
        super(AerosInputFile, self).__init__(fname, blocks)
        self.sep = AerosInputBlock.line_break
        self.log = log
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def write(self):
        super(AerosInputFile, self).write()
        f=open(self.fname, 'a')
        f.write('END')
        f.close()

    def get_output(self, which):
        pass

class AerosInputBlock(InputBlock):
    def __init__(self, name, *args):
        super(AerosInputBlock, self).__init__(name, *args)

    def write(self, fname, indent_level):

        blockvars = vars(self)
        indent = '   '

        f=open(fname,'a+')
        if 'INCLUDE' not in self.name:
            f.write("{0:s}{1:s}\n".format(indent*indent_level, self.name))
        for prop in self.props:
            if prop == 'name': continue
            if blockvars[prop] is None: continue
            s = prop
            if ( hasattr(blockvars[prop], "__getitem__") and
                 type(blockvars[prop]) is not str ):
                for blk in blockvars[prop]:
                    if type(blk) is int:
                        s = "{0:s} {1:d}".format(s, blk)
                    elif type(blk) is float:
                        s = "{0:s} {1:16.12e}".format(s, blk)
                    elif type(blk) is str:
                        s = "{0:s} {1:s}".format(s, blk)
            else:
                if   type(blockvars[prop]) is int:
                    s = "{0:s} {1:d}".format(s, blockvars[prop])
                elif type(blockvars[prop]) is float:
                    s = "{0:s} {1:16.12e}".format(s, blockvars[prop])
                elif type(blockvars[prop]) is str:
                    s = "{0:s} {1:s}".format(s, blockvars[prop])
            f.write("{0:s}\n".format(s))
        f.close()

    @staticmethod
    def line_break(fname):
        f=open(fname,'a'); f.write('*\n'); f.close();

    @staticmethod
    def from_dict(block):
        pass

    @staticmethod
    def from_input_block(block):
        pass
