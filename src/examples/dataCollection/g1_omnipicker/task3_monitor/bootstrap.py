"""Initialize installed ORCA logger before its eager package imports."""
def prepare_logger(directory):
    import importlib.util
    import sys
    from pathlib import Path
    name = 'orca_gym.log.orca_log'
    if name not in sys.modules:
        package = importlib.util.find_spec('orca_gym')
        path = Path(next(iter(package.submodule_search_locations))) / 'log/orca_log.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    sys.modules[name].get_orca_logger(log_dir=str(directory))
