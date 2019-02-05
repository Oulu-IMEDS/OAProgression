import datetime
import os
import pickle
import subprocess


class GlobalKVS(object):
    _instance = None
    _d = dict()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalKVS, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def update(self, tag, value, dtype=None):
        """
        Updates the internal state of the logger.

        Parameters
        ----------
        tag : str
            Tag, of the variable, which we log.
        value : object
            The value to be logged
        dtype :
            Container which is used to store the values under the tag

        Returns
        -------

        """
        if tag not in self._d:
            if dtype is None:
                self._d[tag] = (value, str(datetime.datetime.now()))
            else:
                self._d[tag] = dtype()
        else:
            if isinstance(self._d[tag], list):
                self._d[tag].append((value, str(datetime.datetime.now())))
            elif isinstance(self._d[tag], dict):
                self._d[tag].update((value, str(datetime.datetime.now())))
            else:
                self._d[tag] = (value, str(datetime.datetime.now()))

    def __getitem__(self, tag):
        if not isinstance(self._d[tag], (list, dict)):
            return self._d[tag][0]
        else:
            return self._d[tag]

    def tag_ts(self, tag):
        return self._d[tag]

    def save_pkl(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._d, f)


# Return the git revision as a string
def git_info():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')

        out = _minimal_ext_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        git_branch = out.strip().decode('ascii')
    except OSError:
        return None

    return git_branch, git_revision
