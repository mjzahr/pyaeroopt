from pyaeroopt.util.hpc_util import execute_code

class CodeInterface(object):
    """
    An object to facilitate interfacing with external codes, callable via
    commandline and operate by accepting input file and return output files.

    Data Members
    ------------
    db : SQLAlchemy object
        Database object
    bin : str
        Path to executable
    desc: python object (string, list, tuple, dict, etc)
        Object specifying class instance
    """

    def __init__(self, **kwargs):

        # Extract anticipated input
        self.db   = kwargs.pop('db')   if 'db'   in kwargs else None
        self.bin  = kwargs.pop('bin')  if 'bin'  in kwargs else None
        self.desc = kwargs.pop('desc') if 'desc' in kwargs else None

        # Extract remaining input
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def set_database(self, db):
        self.db = db

    def initialize_database(self, fname):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        engine  = create_engine("sqlite:///{0:s}".format(fname), echo=True)
        self.db = sessionmaker(bind=engine)

    def create_input_file(self, p, desc_ext=None):
        """
        Problem-specific: must be defined by user
        """
        pass

    def check_database(self, infile, desc_ext):
        """
        """
        exist = False
        if self.db is not None:
            exist = self.db.does_exist(self.infile)
        if exist:
            # If instance already exists, update database with external desc
            self.db.update_entry(self.infile, desc_ext)
        return exist

    def execute(self, p, desc_ext=None, hpc=None, make_call=True):
        """
        Check that instance of problem has not been run (if database specified),
        create input file, call executable, and add instance to database (if
        applicible)
        """

        # Create input file and check if instance contained in database
        self.create_input_file(p, desc_ext, self.db)
        exist = self.check_database(self.infile, desc_ext)

        if not exist:
            # If instance does not exist, write input file, call executable,
            # and add to database
            self.infile.write()
            exec_str = hpc.execute_str(self.bin, self.infile.fname)
            execute_code(exec_str, self.infile.log, make_call)
            if self.db is not None: self.db.add_entry(p, self.desc, self.infile)
