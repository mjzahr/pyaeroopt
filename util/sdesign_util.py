import subprocess

tmp_fname  = 'tmp_sdesign_input'
tmp_femesh = 'tmp_sdesign_femesh'

def prepare_directory(fname, femesh):
    """
    """
    subprocess.call('cp {0:s} {1:s}'.format(fname, tmp_fname), shell=True)
    subprocess.call('mv {0:s} {1:s}'.format(femesh, tmp_femesh), shell=True)

def clean_directory(femesh, vmo, der):
    """
    """
    subprocess.call('mv {0:s}.vmo {1:s}'.format(tmp_fname, vmo), shell=True)
    subprocess.call('mv {0:s}.der {1:s}'.format(tmp_fname, der), shell=True)
    subprocess.call('rm {0:s}'.format(tmp_fname), shell=True)
    subprocess.call('mv {0:s} {1:s}'.format(tmp_femesh, femesh), shell=True)
