import numbers

from pyaeroopt.interface import CodeInterface
from pyaeroopt.io        import InputBlock, InputFile
from pyaeroopt.util.hpc_util import execute_str, execute_code
from pyaeroopt.util.sdesign_util import tmp_fname, tmp_femesh
from pyaeroopt.util.sdesign_util import prepare_directory, clean_directory

class Sdesign(CodeInterface):
    """
    An object to facilitate interfacing to AERO-F.
    """
    def __init__(self, **kwargs):

        # Constructor of base class, check fields, and extract anticipated input
        super(Sdesign, self).__init__(**kwargs)

        if self.bin is None: self.bin = os.path.expandvars('$SDESIGN')

    def execute(self, p, desc_ext=None, hpc=None, make_call=True):

        # Create input file and check if instance contained in database
        self.create_input_file(p, desc_ext, self.db)
        exist = self.check_database(self.infile, desc_ext)

        # If instance does not exist, write input file, call executable,
        # and add to database
        if not exist:
            self.infile.write()
            prepare_directory(self.infile.fname, self.femesh)
            exec_str = execute_str(self.bin, tmp_fname)
            execute_code(exec_str, self.infile.log, make_call)
            clean_directory(self.femesh, self.infile.vmo, self.infile.der)
            if self.db is not None: self.db.add_entry(p, self.desc, self.infile)

class SdesignInputFile(InputFile):
    def __init__(self, fname, blocks, log='sdesign.tmp.log', **kwargs):
        super(SdesignInputFile, self).__init__(fname, blocks)
        self.sep = SdesignInputBlock.line_break
        self.log = log
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def write(self):
        InputFile.write(self)
        f=open(self.fname, 'a+')
        f.write('\nEND')
        f.close()

class SdesignInputBlock(InputBlock):
    def __init__(self, name, *args):
        super(SdesignInputBlock, self).__init__(name, *args)

    def write(self, fname, indent_level):

        blockvars = vars(self)
        indent = ''
        f=open(fname,'a+')
        f.write(indent*indent_level+self.name)

        # Formatting for FEMESH vs. rest of blocks
        if self.name == 'FEMESH': f.write('  ')
        else:                     f.write('\n')

        for prop in self.props:
            if prop == 'name': continue

            if self.name == 'DEFINE':
                f.write(indent*(indent_level+1)+prop+' = '
                                                     +str(blockvars[prop])+'\n')
                continue
            if self.name == 'FEMESH':
                f.write(indent*(indent_level+1)+'"'+str(blockvars[prop])+'"\n')
                continue
            if isinstance(blockvars[prop],(str,numbers.Number)):
                f.write(indent*(indent_level+1)+str(blockvars[prop]))
            if isinstance(blockvars[prop],(list,tuple)):
                for x in blockvars[prop]:
                    if isinstance(x,(str,numbers.Number)):
                        f.write(indent*(indent_level+1)+str(x)+'\n')
                    if isinstance(x,(list,tuple)):
                        for y in x:
                            if isinstance(y,(str,numbers.Number)):
                                f.write(indent*(indent_level+1)+str(y)+'    ')
                        f.write('\n')

        f.close()

    @staticmethod
    def line_break(fname):
        f=open(fname,'a'); f.write('\n'); f.close();
