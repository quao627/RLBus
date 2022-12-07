import subprocess, sys, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json, yaml, gc, inspect
from itertools import chain, groupby, islice, product, permutations, combinations
from datetime import datetime
from time import time
from fnmatch import fnmatch
from glob import glob
from tqdm import tqdm
from copy import copy, deepcopy
from collections import OrderedDict, defaultdict, Counter
from colorama import Fore, Style, Back
from gym.spaces import Box, Discrete

import warnings
warnings.filterwarnings('ignore')

from io import StringIO
version = sys.version_info

def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))

def lchain(*args):
    return list(chain(*args))

def lmap(fn, *iterables):
    return [fn(*xs) for xs in zip(*iterables)]

def lif(keep, *x):
    return x if keep else []

def dif(keep, **kwargs):
    return kwargs if keep else {}

def flatten(x):
    return [z for y in x for z in y]

def groupby_(xs, key=None):
    if callable(key):
        key = map(key, xs)
    elif key is None:
        key = xs
    groups = defaultdict(list)
    for k, v in zip(key, xs):
        groups[k].append(v)
    return groups

class Dict(dict if version.major == 3 and version.minor >= 6 else OrderedDict):
    def __add__(self, d):
        return Dict(**self).merge(d)

    def merge(self, *dicts, **kwargs):
        for d in dicts:
            self.update(d)
        self.update(kwargs)
        return self

    def filter(self, keys):
        try: # check for iterable
            keys = set(keys)
            return Dict((k, v) for k, v in self.items() if k in keys)
        except TypeError: # function key
            f = keys
            return Dict((k, v) for k, v in self.items() if f(k, v))

    def map(self, mapper):
        if callable(mapper): # function mapper
            return Dict((k, mapper(v)) for k, v in self.items())
        else: # dictionary mapper
            return Dict((k, mapper[v]) for k, v in self.items())

def parse_dot(d):
    """ Convert dictionary with dot keys to a hierarchical dictionary """
    ks = [(k, v) for k, v in d.items() if '.' in k]
    for k, v in ks:
        del d[k]
        curr = d
        *fronts, back = k.split('.')
        for k_ in fronts:
            curr = curr.setdefault(k_, {})
        curr[back] = v
    return d

def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)

def save_json(path, dict_):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)

def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)

def format_yaml(dict_):
    dict_ = recurse(dict_, lambda x: x._ if isinstance(x, Path) else dict(x) if isinstance(x, Dict) else x)
    return yaml.dump(dict_)

def load_text(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

def save_text(path, string):
    with open(path, 'w') as f:
        f.write(string)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def wget(link, output_dir):
    cmd = 'wget %s -P %s' % (link, output_dir)
    shell(cmd)
    output_path = Path(output_dir) / os.path.basename(link)
    if not output_path.exists(): raise RuntimeError('Failed to run %s' % cmd)
    return output_path

def extract(input_path, output_path=None):
    if input_path[-3:] == '.gz':
        if not output_path:
            output_path = input_path[:-3]
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        raise RuntimeError('Don\'t know file extension for ' + input_path)

def rand_string(length):
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

nexti = nextk = lambda iterable: next(iter(iterable))
nextv = lambda dict: next(iter(dict.values()))
nextkv = lambda dict: next(iter(dict.items()))

def tmux_window(cmd, session='', window='', directory=None):
    def flag(cmds, flag, value):
        if value:
            cmds.extend([flag, value])
        return cmds

    cmds = []
    # if window exists, skip everything
    if window:
        cmds.extend(['tmux', 'list-panes', '-t', '%s:%s' % (session, window)])
        cmds.append('||')

    # else if session exists
    subcmds = ['tmux', 'has-session']
    flag(subcmds, '-t', session)
    subcmds.append('&&')

    # then call new-window
    subcmds.extend(['tmux', 'new-window', '-d'])
    flag(subcmds, '-t', session)
    flag(subcmds, '-n', window)
    flag(subcmds, '-c', directory)
    subcmds.append("'%s'" % cmd)

    cmds.append('(%s)' % ' '.join(subcmds))
    cmds.append('||')

    # else new-session
    cmds.extend(['tmux', 'new-session', '-d'])
    flag(cmds, '-s', session)
    flag(cmds, '-n', window)
    flag(cmds, '-c', directory)

    cmds.append("'%s'" % cmd)
    return ' '.join(cmds)

def ssh(user, host, cmd, key=None, password=None, terminal=False):
    cmds = ['ssh']
    if key is not None:
        cmds.extend(['-i', key])
    if password is not None:
        cmds = ['sshpass', '-p', password] + cmds
    if terminal:
        cmds.append('-t')
    cmds.append('%s@%s' % (user, host))
    cmds.append('"%s"' % cmd)
    return ' '.join(cmds)

def shell(cmd, wait=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    stdout = stdout or subprocess.DEVNULL
    stderr = stderr or subprocess.DEVNULL
    if not isinstance(cmd, str):
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)
    if not wait:
        return process
    out, err = process.communicate()
    return out.decode().rstrip('\n') if out else '', err.decode().rstrip('\n') if err else ''

def terminal_height():
    return int(shell('tput lines')[0])

def terminal_width():
    return int(shell('tput cols')[0])

def git_state(dir=None):
    cwd = os.getcwd()
    dir = dir or shell('git rev-parse --show-toplevel')[0]
    os.chdir(dir)
    status = shell('git status')[0]
    base_commit = shell('git rev-parse HEAD')[0]
    diff = shell('git diff %s' % base_commit)[0]
    os.chdir(cwd)
    return base_commit, diff, status

def attrs(obj):
    for k, v in inspect.getmembers(obj):
        if inspect.isfunction(v) or inspect.ismethod(v):
            print(f'{v.__name__}{inspect.signature(v)}')
        elif not callable(v) and not k.startswith('__'):
            print(k, v)

def source(obj):
    print(inspect.getsource(obj))

def import_module(module_name, module_path):
    import imp
    module = imp.load_source(module_name, module_path)
    return module

def str2num(s):
    try: return int(s)
    except:
        try: return float(s)
        except: return s

def parse_options(defs, *options):
    """
    Each option takes the form of a string keyvalue. Match keyvalue by the following precedence in defs
    defs: {
        keyvalue: {config_key: config_value, ...},
        key: None, # implicitly {key: value}
        key: config_key, # implicitly {config_key: value}
        key: v -> {config_key: config_value, ...},
        ...
    }
    options: [key1value1, key2value2_key3value3, ...]
    """
    options = flatten([x.split('_') for x in options if x])
    name = '_'.join(options)
    kwargs = {}
    for o in options:
        if o in defs:
            kwargs.update(defs[o])
        else:
            k, v = re.match('([a-zA-Z]*)(.*)', o).groups()
            fn_str_none = defs[k]
            if fn_str_none is None:
                kwargs.update({k: v})
            elif isinstance(fn_str_none, str):
                kwargs.update({fn_str_none: v})
            else:
                kwargs.update(fn_str_none(str2num(v)))
    return name, kwargs

def sbatch(cpu=1, gpu=False):
    return f"""#!/bin/sh

#SBATCH -o output-%j.log
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c {cpu}                     # number of cpu per task
{'#SBATCH --gres=gpu:volta:1' if gpu else '#SBATCH --constraint xeon-p8'}

source ~/.bash_profile
"""

def get_time_log_path():
    return datetime.now().isoformat().replace(':', '_').rsplit('.')[0] + '.log'

_log_path = None
def logger(directory=None):
    global _log_path
    if directory and not _log_path:
        from datetime import datetime
        _log_path = Path(directory) / get_time_log_path()
    return log

def log(text, type=None):
    if not type:
        print(text)
    elif type == "WARNING":
        print(Fore.RED + Back.BLACK + type + ": " + text + Style.RESET_ALL)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(text)
            f.write('\n')

def installed(pkg):
    out, err = shell('dpkg -l %s' % pkg)
    if err and err.startswith('dpkg-query: no packages found matching'):
        return False
    return True

def install(pkgs, root):
    root = Path(root)
    old_cwd = os.getcwd()
    self_installed = set()
    os.chdir(root)
    while pkgs:
        pkg = pkgs.pop()
        print('Processing %s' % pkg)
        if installed(pkg) or pkg in self_installed:
            continue
        out, err = shell('apt-cache depends %s' % pkg)
        deps = []
        for x in out.split('\n'):
            x = x.lstrip()
            if x.startswith('Depends:'):
                splits = x.split(' ')
                assert len(splits) == 2
                dep = splits[1]
                if not (dep in self_installed or installed(dep)):
                    deps.append(dep)
        print('Found needed dependencies %s for %s' % (deps, pkg))
        pkgs.extend(deps)
        tmp = Path('tmp')
        shell('mkdir tmp && cd tmp && apt download %s' % pkg)
        for deb in tmp.glob('*.deb'):
            shell('dpkg -x %s .' % deb)
            print('Installing %s with %s' % (pkg, deb))
            self_installed.add(pkg)
        tmp.rm()
    lib = Path('usr/lib')
    real_root = Path('/')
    for x in lib, lib / 'x86_64-linux-gnu':
        brokens = x.lslinks(exist=False)
        for broken in brokens:
            real = real_root / broken._up / os.readlink(broken)
            if real.exists():
                broken.link(real, force=True)
                print('Fixing broken link to be %s -> %s' % (broken, real))
            else:
                print('Could not fix broken link %s' % broken)
    os.chdir(old_cwd)

class Path(str):
    """"""
    @classmethod
    def env(cls, var):
        return Path(os.environ[var])

    def __init__(self, path):
        pass

    def __add__(self, subpath):
        return Path(str(self) + str(subpath))

    def __truediv__(self, subpath):
        return Path(os.path.join(str(self), str(subpath)))

    def __floordiv__(self, subpath):
        return (self / subpath)._

    def ls(self, show_hidden=True, dir_only=False, file_only=False):
        subpaths = [Path(self / subpath) for subpath in os.listdir(self) if show_hidden or not subpath.startswith('.')]
        isdirs = [os.path.isdir(subpath) for subpath in subpaths]
        subdirs = [subpath for subpath, isdir in zip(subpaths, isdirs) if isdir]
        files = [subpath for subpath, isdir in zip(subpaths, isdirs) if not isdir]
        if dir_only:
            return subdirs
        if file_only:
            return files
        return subdirs, files

    def lsdirs(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, dir_only=True)

    def lsfiles(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, file_only=True)

    def lslinks(self, show_hidden=True, exist=None):
        dirs, files = self.ls(show_hidden=show_hidden)
        return [x for x in dirs + files if x.islink() and (
            exist is None or not (exist ^ x.exists()))]

    def glob(self, glob_str):
        return [Path(p) for p in glob(self / glob_str, recursive=True)]

    def re(self, re_pattern):
        """ Similar to .glob but uses regex pattern """
        subpatterns = lmap(re.compile, re_pattern.split('/'))
        matches = []
        dirs, files = self.ls()
        for pattern in subpatterns[:-1]:
            new_dirs, new_files = [], []
            for d in filter(lambda x: pattern.fullmatch(x._name), dirs):
                d_dirs, d_files = d.ls()
                new_dirs.extend(d_dirs)
                new_files.extend(d_files)
            dirs, files = new_dirs, new_files
        return sorted(filter(lambda x: subpatterns[-1].fullmatch(x._name), dirs + files))

    def recurse(self, dir_fn=None, file_fn=None):
        """ Recursively apply dir_fn and file_fn to all subdirs and files in directory """
        if dir_fn is not None:
            dir_fn(self)
        dirs, files = self.ls()
        if file_fn is not None:
            list(map(file_fn, files))
        for dir in dirs:
            dir.recurse(dir_fn=dir_fn, file_fn=file_fn)

    def mk(self):
        os.makedirs(self, exist_ok=True)
        return self

    def dir_mk(self):
        self._up.mk()
        return self

    def rm(self):
        if self.isfile() or self.islink():
            os.remove(self)
        elif self.isdir():
            shutil.rmtree(self)
        return self

    def unlink(self):
        os.unlink(self)
        return self


    def mv(self, dest):
        shutil.move(self, dest)

    def mv_from(self, src):
        shutil.move(src, self)

    def cp(self, dest):
        shutil.copy(self, dest)

    def cp_from(self, src):
        shutil.copy(src, self)

    def link(self, target, force=False):
        if self.lexists():
            if not force:
                return
            else:
                self.rm()
        os.symlink(target, self)

    def exists(self):
        return os.path.exists(self)

    def lexists(self):
        return os.path.lexists(self)

    def isfile(self):
        return os.path.isfile(self)

    def isdir(self):
        return os.path.isdir(self)

    def islink(self):
        return os.path.islink(self)

    def chdir(self):
        os.chdir(self)

    def rel(self, start=None):
        return Path(os.path.relpath(self, start=start))

    def clone(self):
        name = self._name
        match = re.search('__([0-9]+)$', name)
        if match is None:
            base = self + '__'
            i = 1
        else:
            initial = match.group(1)
            base = self[:-len(initial)]
            i = int(initial) + 1
        while True:
            path = Path(base + str(i))
            if not path.exists():
                return path
            i += 1

    @property
    def _(self):
        return str(self)

    @property
    def _real(self):
        return Path(os.path.realpath(os.path.expanduser(self)))

    @property
    def _up(self):
        path = os.path.dirname(self.rstrip('/'))
        if path == '':
            path = os.path.dirname(self._real.rstrip('/'))
        return Path(path)

    @property
    def _name(self):
        return Path(os.path.basename(self))

    @property
    def _stem(self):
        return Path(os.path.splitext(self)[0])

    @property
    def _basestem(self):
        new = self._stem
        while new != self:
            new, self = new._stem, new
        return new

    @property
    def _ext(self):
        return Path(os.path.splitext(self)[1])

    extract = extract
    load_json = load_json
    save_json = save_json
    load_txt = load_sh = load_text
    save_txt = save_sh = save_text
    load_p = load_pickle
    save_p = save_pickle

    def save_bytes(self, bytes):
        with open(self, 'wb') as f:
            f.write(bytes)

    def load_csv(self, index_col=0, **kwargs):
        return pd.read_csv(self, index_col=index_col, **kwargs)

    def save_csv(self, df, float_format='%.5g', **kwargs):
        df.to_csv(self, float_format=float_format, **kwargs)

    def load_npy(self):
        return np.load(self, allow_pickle=True)

    def save_npy(self, obj):
        np.save(self, obj)

    def load_yaml(self):
        with open(self, 'r') as f:
            return yaml.safe_load(f)

    def save_yaml(self, obj):
        obj = recurse(obj, lambda x: x._ if isinstance(x, Path) else dict(x) if isinstance(x, Dict) else x)
        with open(self, 'w') as f:
            yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)

    def load_pth(self):
        return torch.load(self)

    def save_pth(self, obj):
        torch.save(obj, self)

    def load(self):
        return eval('self.load_%s' % self._ext[1:])()

    def save(self, obj):
        return eval('self.save_%s' % self._ext[1:])(obj)

    def replace_txt(self, replacements, dst=None):
        content = self.load_txt()
        for k, v in replacements.items():
            content = content.replace(k, v)
        (dst or self).save_txt(content)

    def update_dict(self, updates={}, vars=[], unvars=[], dst=None):
        d = self.load()
        for k in vars:
            d[k] = True
        for k in unvars:
            d.pop(k, None)
        d.update(updates)
        (dst or self).save(d)

    def torch_strip(self, dst):
        self.update_dict(unvars=['opt', 'step'], dst=dst)

    def wget(self, link):
        if self.isdir():
            return Path(wget(link, self))
        raise ValueError('Path %s needs to be a directory' % self)

    def replace(self, old, new=''):
        return Path(super().replace(old, new))

    def search(self, pattern):
        return re.search(pattern, self)

    def search_pattern(self, pattern):
        return self.search(pattern).group()

    def search_groups(self, pattern):
        return self.search(pattern).groups()

    def search_group(self, pattern):
        return self.search_groups(pattern)[0]

    def findall(self, pattern):
        return re.findall(pattern, self)

class Namespace(Dict):
    def __init__(self, *args, **kwargs):
        self.var(*args, **kwargs)

    def var(self, *args, **kwargs):
        kvs = Dict()
        for a in args:
            if isinstance(a, str):
                kvs[a] = True
            else: # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.update(kvs)
        return self

    def unvar(self, *args):
        for a in args:
            self.pop(a)
        return self

    def setdefaults(self, *args, **kwargs):
        args = [a for a in args if a not in self]
        kwargs = {k: v for k, v in kwargs.items() if k not in self}
        return self.var(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            self.__getattribute__(key)

    def __setattr__(self, key, value):
        self[key] = value

##### Functions for compute

using_ipython = True
try:
    _ = get_ipython().__class__.__name__
except NameError:
    using_ipython = False

try:
    import numpy as np

    import pandas as pd
    def _sel(self, col, value):
        if isinstance(value, list):
            return self[self[col].isin(value)]
        return self[self[col] == value]
    pd.DataFrame.sel = _sel

    import scipy.stats
    import scipy as sp
    from scipy.stats import pearsonr as pearson, spearmanr as spearman, kendalltau

    if not using_ipython:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    arrayf = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.float32)
    arrayl = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.long)
    arrayb = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.bool)
    arrayo = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=object)

except ImportError:
    pass

from sklearn.metrics import roc_auc_score as auroc, average_precision_score as auprc, roc_curve as roc, precision_recall_curve as prc, accuracy_score as accuracy

def split(x, sizes):
    return np.split(x, np.cumsum(sizes[:-1]))

def recurse(x, fn):
    if isinstance(x, dict):
        return type(x)((k, recurse(v, fn)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(recurse(v, fn) for v in x)
    return fn(x)

def from_numpy(x):
    def helper(x):
        if type(x).__module__ == np.__name__:
            if isinstance(x, np.ndarray):
                return recurse(list(x), helper)
            return np.asscalar(x)
        return x
    return recurse(x, helper)

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def gsmooth(y, sigma):
    from scipy.ndimage.filters import gaussian_filter1d
    return gaussian_filter1d(y, sigma=sigma)

def normalize(x, eps=1e-8):
    return (x - x.mean()) / x.std()

def inverse_map(arr):
    inv_map = np.zeros(len(arr))
    inv_map[arr] = np.arange(len(arr))
    return inv_map

def pad_arrays(arrs, value):
    max_len = max(len(x) for x in arrs)
    return np.array([np.concatenate([x, np.full(max_len - len(x), value)]) for x in arrs])

def sorted_segment_maps(segments):
    r = Namespace()
    r.segment_idxs = np.argsort(segments)
    starts = np.cumsum(segments) - segments
    starts = starts[r.segment_idxs]
    r.segments = segments[r.segment_idxs]

    r.unit_idxs = np.array([i for s, length in zip(starts, r.segments) for i in range(s, s + length)])
    r.unit_idxs_r = inverse_map(r.unit_idxs)
    r.segment_uniques, r.segment_blocks, r.segment_counts = zip(*((seg, sum(segs), len(list(segs))) for seg, segs in groupby(r.segments)))
    return r

def reindex(df, order=None, rename=None, level=[], axis=0, squeeze=True):
    assert axis in [0, 1]
    if not isinstance(level, list):
        if order is not None: order = [order]
        if rename is not None: rename = [rename]
        level = [level]
    if order is None: order = [[]] * len(level)
    if rename is None: rename = [{}] * len(level)
    assert len(level) == len(rename) == len(order)
    multiindex = df.index
    if axis == 1:
        multiindex = df.columns
    for i, (o, lev) in enumerate(zip(order, level)):
        if len(o) == 0:
            seen = set()
            new_o = []
            for k in multiindex.get_level_values(lev):
                if k in seen: continue
                new_o.append(k)
                seen.add(k)
            order[i] = new_o
    assert len(set(level) - set(multiindex.names)) == 0, 'Levels %s not in index %s along axis %s' % (level, axis, multiindex.names)
    lev_order = dict(zip(level, order))
    level_map = {}
    for lev in multiindex.names:
        if lev in level:
            level_map[lev] = { name : i for i, name in enumerate(lev_order[lev]) }
        else:
            index_map = {}
            for x in multiindex.get_level_values(lev):
                if x in index_map: continue
                index_map[x] = len(index_map)
            level_map[lev] = index_map
    tuples = list(multiindex)
    def get_numerical(tup):
        return tuple(level_map[lev][t] for t, lev in zip(tup, multiindex.names))
    filtered_tuples = [tup for tup in tuples if all(t in level_map[lev] for t, lev in zip(tup, multiindex.names))]
    new_tuples = sorted(filtered_tuples, key=get_numerical)
    lev_rename = dict(zip(level, rename))
    renamed_tuples = [tuple(lev_rename.get(lev, {}).get(t, t) for t, lev in zip(tup, multiindex.names)) for tup in new_tuples]
    new_index = pd.MultiIndex.from_tuples(new_tuples, names=multiindex.names)
    renamed_index = pd.MultiIndex.from_tuples(renamed_tuples, names=multiindex.names)
    if squeeze:
        single_levels = [i for i, level in enumerate(renamed_index.levels) if len(level) == 1]
        renamed_index = renamed_index.droplevel(single_levels)
    if axis == 0:
        new_df = df.loc[new_index]
        new_df.index = renamed_index
    else:
        new_df = df.loc[:, new_index]
        new_df.columns = renamed_index
    return new_df

def get_gpu_info(ssh_fn=lambda x: x):
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,nounits'))
    nvidia_str = nvidia_str.replace('[Not Supported]', '100').replace(', ', ',')
    nvidia_str_io = StringIO(nvidia_str)

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    devices_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices_str:
        devices = list(map(int, devices_str.split(',')))
        gpu_df = gpu_df.loc[devices]
        gpu_df.index = gpu_df.index.map({k: i for i, k in enumerate(devices)})

    out_df = pd.DataFrame(index=gpu_df.index)
    out_df['memory_total'] = gpu_df['memory.total [MiB]']
    out_df['memory_used'] = gpu_df['memory.used [MiB]']
    out_df['memory_free'] = out_df['memory_total'] - out_df['memory_used']
    out_df['utilization'] = gpu_df['utilization.gpu [%]'] / 100
    out_df['utilization_free'] = 1 - out_df['utilization']
    return out_df

def get_process_gpu_info(pid=None, ssh_fn=lambda x: x):
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-compute-apps=pid,gpu_name,used_gpu_memory --format=csv,nounits'))
    nvidia_str_io = StringIO(nvidia_str.replace(', ', ','))

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    if pid is None:
        return gpu_df
    if pid == -1:
        pid = os.getpid()
    return gpu_df.loc[pid]

##### torch functions

try:

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    def to_torch(x, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        def helper(x):
            if x is None:
                return None
            elif isinstance(x, torch.Tensor):
                return x.to(device=device, **kwargs)
            elif np.isscalar(x):
                return x
            return torch.from_numpy(x).to(device=device, **kwargs)
        return recurse(x, helper)

    def from_torch(t, force_scalar=False):
        def helper(t):
            if not isinstance(t, torch.Tensor):
                return t
            x = t.detach().cpu().numpy()
            if force_scalar and (x.size == 1 or np.isscalar(x)):
                return np.asscalar(x)
            return x
        return recurse(t, helper)

    def count_params(network, requires_grad=False):
        return sum(p.numel() for p in network.parameters() if not requires_grad or p.requires_grad)

    def report_memory(device=None, max=False):
        if device:
            device = torch.device(device)
            if max:
                alloc = torch.cuda.max_memory_allocated(device=device)
            else:
                alloc = torch.cuda.memory_allocated(device=device)
            alloc /=  1024 ** 2
            print('%.3f MBs' % alloc)
            return alloc

        numels = Counter()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(type(obj), obj.size())
                numels[obj.device] += obj.numel()
        print()
        for device, numel in sorted(numels.items()):
            print('%s: %s elements, %.3f MBs' % (str(device), numel, numel * 4 / 1024 ** 2))

    def clear_gpu_memory():
        gc.collect()
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                obj.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    class Reshape(nn.Module):
        def __init__(self, *shape, split=None, merge=None):
            super(Reshape, self).__init__()
            self.shape = shape
            self.split = split
            self.merge = merge

        def forward(self, input):
            if self.split is None and self.merge is None:
                return input.reshape(*self.shape)
            in_shape = input.shape

    class Transpose(nn.Module):
        def __init__(self, dim0, dim1):
            super(Transpose, self).__init__()
            self.dim0 = dim0
            self.dim1 = dim1

        def forward(self, input):
            return input.transpose(self.dim0, self.dim1)

    class Permute(nn.Module):
        def __init__(self, *dims):
            super(Permute, self).__init__()
            self.dims = dims

        def forward(self, input):
            return input.permute(*self.dims)

    class CausalConv1d(nn.Module):
        def __init__(self, in_depth, out_depth, kernel_size, dilation=1, stride=1, groups=1):
            super(CausalConv1d, self).__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_depth, out_depth, kernel_size, stride=stride, dilation=dilation, groups=groups)

        def forward(self, x, pad=True):
            if pad:
                x = F.pad(x, (self.padding, 0))
            return self.conv(x)

    class CausalMaxPool1d(nn.Module):
        def __init__(self, kernel_size, dilation=1, stride=1):
            super(CausalMaxPool1d, self).__init__()
            self.padding = (kernel_size - 1) * dilation
            self.pool = nn.MaxPool1d(kernel_size, stride=stride, dilation=dilation)

        def forward(self, x, pad=True):
            if pad:
                x = F.pad(x, (self.padding, 0))
            return self.pool(x)

except ImportError:
    pass

try:
    from apex import amp
except ImportError:
    pass

def main_only(method):
    def wrapper(self, *args, **kwargs):
        if self.main:
            return method(self, *args, **kwargs)
    return wrapper

class Config(Namespace):
    def __init__(self, res, *args, **kwargs):
        self.res = Path(res)._real
        super(Config, self).__init__()
        self.load()
        self.var(*args, **kwargs)
        self.setdefaults(
            name=self.res._real._name,
            main=True,
            logger=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            debug=False,
            opt_level='O0',
            disable_amp=False
        )

    def __repr__(self):
        return format_yaml(dict(self))

    def __hash__(self):
        return hash(repr(self))

    @property
    def path(self):
        return self.res / 'config.yaml'

    def load(self):
        if self.path.exists():
            for k, v in self.path.load().items():
                self[k] = v
        return self

    never_save = {'res', 'name', 'main', 'logger', 'distributed', 'parallel', 'device', 'steps', 'debug'}
    @property
    def attrs_save(self):
        return {k: v for k, v in self.items() if k not in self.never_save}

    def save(self, force=False):
        if force or not self.path.exists():
            self.res.mk()
            self.path.save(from_numpy(self.attrs_save))
        return self

    @classmethod
    def from_args(cls, *globals_locals):
        import argparse
        parser = argparse.ArgumentParser(description='Model arguments')
        parser.add_argument('res', type=Path, help='Result directory')
        parser.add_argument('kwargs', nargs='*', help='Extra arguments that goes into the config')

        args = parser.parse_args()

        kwargs = {}
        for kv in args.kwargs:
            splits = kv.split('=')
            if len(splits) == 1:
                v = True
            else:
                v = splits[1]
                try:
                    v = eval(v, *globals_locals)
                except (SyntaxError, NameError):
                    pass
            kwargs[splits[0]] = v

        return cls(args.res, **kwargs).save()

    def try_save_commit(self, base_dir=None):
        base_commit, diff, status = git_state(base_dir)

        save_dir = (self.res / 'commit').mk()
        (save_dir / 'hash.txt').save(base_commit)
        (save_dir / 'diff.txt').save(diff)
        (save_dir / 'status.txt').save(status)
        return self

    @main_only
    def log(self, text, type=None):
        logger(self.res if self.logger else None)(text, type)

    ### Train result saving ###

    @property
    def train_results(self):
        return self.res / 'train_results.csv'

    def load_train_results(self):
        if self.train_results.exists():
            return pd.read_csv(self.train_results, index_col=0)
        return None

    @main_only
    def save_train_results(self, results):
        results.to_csv(self.train_results, float_format='%.6g')

    ### Set stopped early ###

    @property
    def stopped_early(self):
        return self.res / 'stopped_early'

    @main_only
    def set_stopped_early(self):
        self.stopped_early.save_txt('')

    ### Set training state ###

    @property
    def training(self):
        return self.res / 'is_training'

    @main_only
    def set_training(self, is_training):
        if is_training:
            if self.main and self.training.exists():
                self.log('Another training is found, continue (yes/n)?')
                ans = input('> ')
                if ans != 'yes':
                    exit()
            self.training.save_txt('')
        else:
            self.training.rm()

    ### Model loading ###

    def init_model(self, net, opt=None, step='max', train=True):
        if train:
            assert not self.training.exists(), 'Training already exists'
        # configure parallel training
        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.n_gpus = 0 if self.device == 'cpu' else 1 if self.device.startswith('cuda:') else len(get_gpu_info()) if devices is None else len(devices.split(','))
        can_parallel = self.n_gpus > 1
        self.setdefaults(distributed=can_parallel) # use distributeddataparallel
        self.setdefaults(parallel=can_parallel and not self.distributed) # use dataparallel
        self.local_rank = 0
        self.world_size = 1 # number of processes
        if self.distributed:
            self.local_rank = int(os.environ['LOCAL_RANK']) # rank of the current process
            self.world_size = int(os.environ['WORLD_SIZE'])
            assert self.world_size == self.n_gpus
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.main = self.local_rank == 0

        net.to(self.device)
        if train and not self.disable_amp:
            # configure mixed precision
            net, opt = amp.initialize(net, opt, opt_level=self.opt_level, loss_scale=self.get('loss_scale'), verbosity=0 if self.opt_level == 'O0' else 1)
        step = self.set_state(net, opt=opt, step=step)

        if self.distributed:
            import apex
            net = apex.parallel.DistributedDataParallel(net)
        elif self.parallel:
            net = nn.DataParallel(net)

        if train:
            net.train()
            return net, opt, step
        else:
            net.eval()
            return net, step

    def load_model(self, step='best', train=False):
        '''
        step can be 'best', 'max', an integer, or None
        '''
        model = import_module('model', str(self.model))
        net = model.get_net(self)
        opt = model.get_opt(self, net) if train else None
        return self.init_model(net, opt=opt, step=step, train=train)

    @property
    def models(self):
        return (self.res / 'models').mk()

    def model_save(self, step):
        return self.models / ('model-%s.pth' % step)

    def model_step(self, path):
        m = re.match('.+/model-(\d+)\.pth', path)
        if m:
            return int(m.groups()[0])

    @property
    def model_best(self):
        return self.models / 'best_model.pth'

    @main_only
    def link_model_best(self, model_save):
        self.model_best.rm().link(Path(model_save).rel(self.models))

    def get_saved_model_steps(self):
        _, save_paths = self.models.ls()
        if len(save_paths) == 0:
            return []
        return sorted([x for x in map(self.model_step, save_paths) if x is not None])

    def set_state(self, net, opt=None, step='max', path=None):
        state = self.load_state(step=step, path=path)
        if state is None:
            return 0
        if self.get('append_module_before_load'):
            state['net'] = OrderedDict(('module.' + k, v) for k, v in state['net'].items())
        net.load_state_dict(state['net'])
        if opt:
            if 'opt' in state:
                opt.load_state_dict(state['opt'])
            else:
                self.log('No state for optimizer to load')
        if 'amp' in state and self.opt_level != 'O0':
            amp.load_state_dict(state['amp'])
        return state.get('step', 0)

    @main_only
    def get_state(self, net, opt, step):
        try:
            net_dict = net.module.state_dict()
        except AttributeError:
            net_dict = net.state_dict()
        state = dict(step=step, net=net_dict, opt=opt.state_dict())
        try:
            state['amp'] = amp.state_dict()
        except:
            pass
        return to_torch(state, device='cpu')

    def load_state(self, step='max', path=None):
        '''
        step: best, max, integer, None if path is specified
        path: None if step is specified
        '''
        if path is None:
            if step == 'best':
                path = self.model_best
            else:
                if step == 'max':
                    steps = self.get_saved_model_steps()
                    if len(steps) == 0:
                        return None
                    step = max(steps)
                path = self.model_save(step)
        save_path = Path(path)
        if save_path.exists():
            return to_torch(torch.load(save_path), device=self.device)
        return None

    @main_only
    def save_state(self, step, state, clean=True, link_best=False):
        save_path = self.model_save(step)
        if save_path.exists():
            return save_path
        torch.save(state, save_path)
        self.log('Saved model %s at step %s' % (save_path, step))
        if clean and self.get('max_save'):
            self.clean_models(keep=self.max_save)
        if link_best:
            self.link_model_best(save_path)
            self.log('Linked %s to new saved model %s' % (self.model_best, save_path))
        return save_path

    ### Utility methods for manipulating experiments ###

    def clone(self):
        return self._clone().save()

    def clone_(self):
        return self.cp_(self.res._real.clone())

    def cp(self, path, *args, **kwargs):
        return self.cp_(path, *args, **kwargs).save()

    def cp_(self, path, *args, **kwargs):
        '''
        path: should be absolute or relative to self.res._up
        '''
        attrs = self.attrs_save
        for a in args:
            kwargs[a] = True
        kwargs = {k: v for k, v in kwargs.items() if v != attrs.get(k)}

        merged = attrs.copy()
        merged.update(kwargs)

        if os.path.isabs(path):
            new_res = path
        else:
            new_res = self.res._up / path
        return Config(new_res).var(**merged)

    @classmethod
    def load_all(cls, *directories, df=False, kwargs={}):
        configs = []
        def dir_fn(d):
            c = Config(d, **kwargs)
            if not c.path.exists():
                return
            configs.append(c.load())
        for d in map(Path, directories):
            d.recurse(dir_fn)
        if not df:
            return configs
        config_map = {c: c.attrs_save for c in configs}
        return pd.DataFrame(config_map).T.fillna('')

    @classmethod
    def clean(self, cls, *directories):
        configs = cls.load_all(*directories)
        for config in configs:
            if not (config.train_results.exists() or len(config.models.ls()[1]) > 0):
                config.res.rm()
                self.log('Removed %s' % config.res)

    @main_only
    def clean_models(self, keep=5):
        model_steps = self.get_saved_model_steps()
        delete = len(model_steps) - keep
        keep_paths = [self.model_best._real, self.model_save(model_steps[-1])._real]
        for e in model_steps:
            if delete <= 0:
                break
            path = self.model_save(e)._real
            if path in keep_paths:
                continue
            path.rm()
            delete -= 1
            self.log('Removed model %s' % path.rel(self.res))


def discount(x, gamma):
    if isinstance(x, torch.Tensor):
        n = x.size(0)
        return F.conv1d(F.pad(x, (0, n - 1)).view(1, 1, -1), gamma ** torch.arange(n, dtype=x.dtype).view(1, 1, -1)).view(-1)
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def explained_variance(y_pred, y_true):
    if not len(y_pred):
        return np.nan
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

class RunningStats:
    '''
    Tracks first and second moments (mean and variance) of a streaming time series
    https://github.com/joschu/modular_rl
    http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self):
        self.n = 0
        self.mean = 0
        self._nstd = 0

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean
            self.mean = old_mean + (x - old_mean) / self.n
            self._nstd = self._nstd + (x - old_mean) * (x - self.mean)
    @property
    def var(self):
        return self._nstd / (self.n - 1) if self.n > 1 else np.square(self.mean)
    @property
    def std(self):
        return np.sqrt(self.var)

class NamedArrays(dict):
    """
    Data structure for keeping track of a dictionary of arrays (used for rollout information)
    e.g. {'reward': [...], 'action': [...]}
    """
    def __init__(self, dict_of_arrays={}, **kwargs):
        kwargs.update(dict_of_arrays)
        super().__init__(kwargs)

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __getitem__(self, k):
        if isinstance(k, (slice, int, np.ndarray, list)):
            return type(self)((k_, arr[k]) for k_, arr in self.items())
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, (slice, int, np.ndarray, list)):
            if isinstance(v, dict):
                for vk, vv in v.items():
                    self[vk][k] = vv
            else:
                for k_, arr in self.items():
                    arr[k] = v
        else:
            super().__setitem__(k, v)

    def append(self, *args, **kwargs):
        for k, v in itertools.chain(args, kwargs.items()):
            if isinstance(v, dict):
                self.setdefault(k, type(self)()).append(**v)
            else:
                self.setdefault(k, []).append(v)

    def extend(self, *args, **kwargs):
        for k, v in itertools.chain(args, kwargs.items()):
            if isinstance(v, dict):
                self.setdefault(k, type(self)()).extend(**v)
            else:
                self.setdefault(k, []).extend(v)

    def to_array(self, inplace=True, dtype=None, concat=False):
        return self.apply(np.concatenate if concat else lambda x: np.asarray(x, dtype=dtype), inplace)

    def to_torch(self, dtype=None, device=None):
        for k, v in self.items():
            if isinstance(v, list):
                v = np.asarray(v, dtype=dtype)
            if isinstance(v, NamedArrays):
                v.to_torch(dtype, device)
            else:
                if v.dtype == object:
                    self[k] = [torch.tensor(np.ascontiguousarray(x), device=device) for x in v]
                else:
                    self[k] = torch.tensor(np.ascontiguousarray(v), device=device)
        return self

    def trim(self):
        min_len = len(self)
        for k, v in self.items():
            self[k] = v[:min_len]

    def __len__(self):
        return len(self.keys()) and min(len(v) for v in self.values())

    def filter(self, *args):
        return type(self)((k, v) for k, v in self.items() if k in args)

    def iter_minibatch(self, n_minibatches=None, concat=False, device='cpu'):
        if n_minibatches in [1, None]:
            yield slice(None), self.to_array(inplace=False, concat=concat).to_torch(device=device)
        else:
            for idxs in np.array_split(np.random.permutation(len(self)), n_minibatches):
                na = type(self)((k, v[idxs]) for k, v in self.items())
                yield idxs, na.to_array(inplace=False, concat=concat).to_torch(device=device)

    def apply(self, fn, inplace=True):
        if inplace:
            for k, v in self.items():
                if isinstance(v, NamedArrays):
                    v.apply(fn)
                else:
                    self[k] = fn(v)
            return self
        else:
            return type(self)((k, v.apply(fn, inplace=False) if isinstance(v, NamedArrays) else fn(v)) for k, v in self.items())

    def __getstate__(self):
        return dict(**self)

    def __setstate__(self, d):
        self.update(d)

    @classmethod
    def concat(cls, named_arrays, fn=None):
        named_arrays = list(named_arrays)
        if not len(named_arrays):
            return cls()
        def concat(xs):
            """
            Common error with np.concatenate: conside arrays a and b, both of which are lists of arrays. If a contains irregularly shaped arrays and b contains arrays with the same shape, the numpy will treat b as a 2D array, and the concatenation will fail. Solution: use flatten instead of np.concatenate for lists of arrays
            """
            try: return np.concatenate(xs)
            except: return flatten(xs)
        get_concat = lambda v: v.concat if isinstance(v, NamedArrays) else fn or (torch.cat if isinstance(v, torch.Tensor) else concat)
        return cls((k, get_concat(v)([x[k] for x in named_arrays])) for k, v in named_arrays[0].items())

class Dist:
    """ Distribution interface """
    def __init__(self, inputs):
        self.inputs = inputs

    def sample(self, shape=torch.Size([])):
        return self.dist.sample(shape)

    def argmax(self):
        raise NotImplementedError

    def logp(self, actions):
        return self.dist.log_prob(actions)

    def kl(self, other):
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    def entropy(self):
        return self.dist.entropy()

    def __getitem__(self, idx):
        return type(self)(self.inputs[idx])

class CatDist(Dist):
    """ Categorical distribution (for discrete action spaces) """
    def __init__(self, inputs):
        super().__init__(inputs)
        self.dist = torch.distributions.categorical.Categorical(logits=inputs)

    def argmax(self):
        return self.dist.probs.argmax(dim=-1)

class DiagGaussianDist(Dist):
    """ Diagonal Gaussian distribution (for continuous action spaces) """
    def __init__(self, inputs):
        super().__init__(inputs)
        self.mean, self.log_std = torch.chunk(inputs, 2, dim=-1)
        self.std = self.log_std.exp()
        self.dist = torch.distributions.normal.Normal(self.mean, self.std)

    def argmax(self):
        return self.dist.mean

    def logp(self, actions):
        return super().logp(actions).sum(dim=-1)

    def kl(self, other):
        return super().kl(other).squeeze(dim=-1)

    def entropy(self):
        return super().entropy().squeeze(dim=-1)

def build_dist(space):
    """
    Build a nested distribution
    """
    if isinstance(space, Box):
        class DiagGaussianDist_(DiagGaussianDist):
            model_output_size = np.prod(space.shape) * 2
        return DiagGaussianDist_
    elif isinstance(space, Discrete):
        class CatDist_(CatDist):
            model_output_size = space.n
        return CatDist_

    assert isinstance(space, dict) # Doesn't support lists at the moment since there's no list equivalent of NamedArrays that allows advanced indexing
    names, subspaces = zip(*space.items())
    to_list = lambda x: [x[name] for name in names]
    from_list = lambda x: NamedArrays(zip(names, x))
    subdist_classes = [build_dist(subspace) for subspace in subspaces]
    subsizes = [s.model_output_size for s in subdist_classes]
    class Dist_(Dist):
        model_output_size = sum(subsizes)
        def __init__(self, inputs):
            super().__init__(inputs)
            self.dists = from_list(cl(x) for cl, x in zip(subdist_classes, inputs.split(subsizes, dim=-1)))

        def sample(self, shape=torch.Size([])):
            return from_list([dist.sample(shape) for dist in to_list(self.dists)])

        def argmax(self):
            return from_list([dist.argmax() for dist in to_list(self.dists)])

        def logp(self, actions):
            return sum(dist.logp(a) for a, dist in zip(to_list(actions), to_list(self.dists)))

        def kl(self, other):
            return sum(s.kl(o) for s, o in zip(to_list(self.dists), to_list(other.dists)))

        def entropy(self):
            return sum(dist.entropy() for dist in to_list(self.dists))
    return Dist_

def build_fc(input_size, *sizes_and_modules):
    """
    Build a fully connected network
    """
    layers = []
    str_map = dict(relu=nn.ReLU(inplace=True), tanh=nn.Tanh(), sigmoid=nn.Sigmoid(), flatten=nn.Flatten(), softmax=nn.Softmax())
    for x in sizes_and_modules:
        if isinstance(x, (int, np.integer)):
            input_size, x = x, nn.Linear(input_size, x)
        if isinstance(x, str):
            x = str_map[x]
        layers.append(x)
    return nn.Sequential(*layers)

class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c.setdefaults(layers=[64, 'tanh', 64, 'tanh'], weight_scale='default', weight_init='orthogonal')
        layers = c.layers
        if isinstance(layers, list):
            layers = Namespace(s=[], v=layers, p=layers)
        s_sizes = [c.observation_space.shape[0], *layers.s]

        self.shared = build_fc(*s_sizes)

        self.p_head = build_fc(s_sizes[-1], *layers.p, c.model_output_size)
        self.sequential_init(self.p_head, 'policy')
        self.v_head = None
        if c.use_critic:
            self.v_head = build_fc(s_sizes[-1], *layers.v, 1)
            self.sequential_init(self.v_head, 'value')

    def sequential_init(self, seq, key):
        c = self.c
        linears = [m for m in seq if isinstance(m, nn.Linear)]
        for i, m in enumerate(linears):
            if isinstance(c.weight_scale, (int, float)):
                scale = c.weight_scale
            elif isinstance(c.weight_scale, (list, tuple)):
                scale = c.weight_scale[i]
            elif isinstance(c.weight_scale, (dict)):
                scale = c.weight_scale[key][i]
            else:
                scale = 0.01 if m == linears[-1] else 1
            if c.weight_init == 'normc': # normalize along input dimension
                weight = torch.randn_like(m.weight)
                m.weight.data = weight * scale / weight.norm(dim=1, keepdim=True)
            elif c.weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=scale)
            elif c.weight_init == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=scale)
            nn.init.zeros_(m.bias)

    def forward(self, inp, value=False, policy=False, argmax=None):
        s = self.shared(inp)
        pred = Namespace()
        if value and self.v_head:
            pred.value = self.v_head(s).view(-1)
        if policy or argmax is not None:
            pred.policy = self.p_head(s)
            if argmax is not None:
                dist = self.c.dist_class(pred.policy)
                pred.action = dist.argmax() if argmax else dist.sample()
        return pred

def calc_adv(reward, gamma, value_=None, lam=None):
    """
    Calculate advantage with TD-lambda
    """
    if value_ is None:
        return discount(reward, gamma), None # TD(1)
    if isinstance(reward, list):
        reward, value_ = map(np.array, (reward, value_))
    assert value_.ndim == reward.ndim == 1, f'Value and reward be one dimensional, but got {value_.shape} and {reward.shape} respectively'
    assert value_.shape[0] - reward.shape[0] in [0, 1], f'Value\'s shape can be at most 1 bigger than reward\'s shape, but got {value_.shape} and {reward.shape} respectively'

    if value_.shape[0] == reward.shape[0]:
        delta = reward - value_
        delta[:-1] += gamma * value_[1:]
    else:
        delta = reward + gamma * value_[1:] - value_[:-1]
    adv = discount(delta, gamma * lam)
    ret = value_[:len(adv)] + adv
    return ret, adv

def calc_adv_multi_agent(id_, reward, gamma, value_=None, lam=None):
    """
    Calculate advantage with TD-lambda for multiple agents

    id_ and value_ include the last time step, reward does not include the last time step
    id_ should be something that pandas.Series.groupby works on
    id_, reward, and value_ should be flat arrays with "shape" n_steps * n_agent_per_step
    """
    n_id = len(reward) # number of ids BEFORE the last time step
    ret = np.empty((n_id,), dtype=np.float32)
    adv = ret.copy()
    for _, group in pd.Series(id_).groupby(id_):
        idxs = group.index
        value_i_ = None if value_ is None else value_[idxs]
        if idxs[-1] >= n_id:
            idxs = idxs[:-1]
        ret[idxs], adv[idxs] = calc_adv(reward=reward[idxs], gamma=gamma, value_=value_i_, lam=lam)
    return ret, adv


class Algorithm:
    """
    RL algorithm interface
    """
    def __init__(self, c):
        self.c = c.setdefaults(normclip=None, use_critic=True, lam=1, batch_concat=False, device='cpu')

    def on_step_start(self):
        return {}

    def optimize(self, rollouts):
        raise NotImplementedError

    def value_loss(self, v_pred, ret, v_start=None, mask=None):
        c = self.c
        mask = slice(None) if mask is None else mask
        unclipped = (v_pred - ret) ** 2
        if v_start is None or c.vclip is None: # no value clipping
            return unclipped[mask].mean()
        clipped_value = v_start + (v_pred - v_start).clamp(-c.vclip, c.vclip)
        clipped = (clipped_value - ret) ** 2
        return torch.max(unclipped, clipped)[mask].mean() # no gradient if larger

    def step_loss(self, loss):
        c = self.c
        c._opt.zero_grad()
        if torch.isnan(loss):
            import q 
            q.d()
            raise RuntimeError('Encountered nan loss during training')
        loss.backward()
        if c.normclip:
            torch.nn.utils.clip_grad_norm_(c._model.parameters(), c.normclip)
        c._opt.step()

class PPO(Algorithm):
    def __init__(self, c):
        super().__init__(c.setdefaults(use_critic=True, n_gds=30, pclip=0.3, vcoef=1, vclip=1, klcoef=0.2, kltarg=0.01, entcoef=0))

    def on_step_start(self):
        stats = dict(klcoef=self.c.klcoef)
        if self.c.entcoef:
            stats['entcoef'] = self.entcoef
        return stats

    @property
    def entcoef(self):
        c = self.c
        return c.schedule(c.entcoef, c.get('ent_schedule'))

    def optimize(self, rollouts):
        c = self.c
        batch = rollouts.filter('obs', 'policy', 'action', 'pg_obj', 'ret', *lif(c.use_critic, 'value', 'adv'))
        value_warmup = c._i < c.get('n_value_warmup', 0)

        for i_gd in range(c.n_gds):
            batch_stats = []
            for idxs, mb in batch.iter_minibatch(c.get('n_minibatches'), concat=c.batch_concat, device=c.device):
                if not len(mb):
                    continue
                start_dist = c.dist_class(mb.policy)
                start_logp = start_dist.logp(mb.action)
                if 'pg_obj' not in batch:
                    mb.pg_obj = mb.adv if c.use_critic else mb.ret
                pred = c._model(mb.obs, value=True, policy=True)
                curr_dist = c.dist_class(pred.policy)
                p_ratio = (curr_dist.logp(mb.action) - start_logp).exp()

                pg_obj = mb.pg_obj
                if c.adv_norm:
                    pg_obj = normalize(pg_obj)

                policy_loss = -torch.min(
                    pg_obj * p_ratio,
                    pg_obj * p_ratio.clamp(1 - c.pclip, 1 + c.pclip) # no gradient if larger
                ).mean()

                kl = start_dist.kl(curr_dist).mean()
                entropy = curr_dist.entropy().mean()

                loss = policy_loss + c.klcoef * kl - self.entcoef * entropy
                stats = dict(
                    policy_loss=policy_loss, kl=kl, entropy=entropy
                )

                if value_warmup:
                    loss = loss.detach()

                if c.use_critic:
                    value_mask = mb.obs.get('value_mask') if isinstance(mb.obs, dict) else None
                    value_loss = self.value_loss(pred.value, mb.ret, v_start=mb.value, mask=value_mask)
                    loss += c.vcoef * value_loss
                    stats['value_loss'] = value_loss
                self.step_loss(loss)
                batch_stats.append(from_torch(stats))
            c.log_stats(pd.DataFrame(batch_stats).mean(axis=0), ii=i_gd, n_ii=c.n_gds)

        if c.klcoef:
            kl = from_torch(kl)
            if kl > 2 * c.kltarg:
                c.klcoef *= 1.5
            elif kl < 0.5 * c.kltarg:
                c.klcoef *= 0.5
