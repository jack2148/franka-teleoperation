"""Explicit SDK-dependent gate; does not open a device."""
import os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_native_actual_hdapi_callback_has_one_frame_on_start_fault_and_exit(tmp_path):
    runtime=Path(os.environ.get('GEOMAGIC_LOCAL',str(Path.home()/'.local/geomagic')))
    binary=tmp_path/'native_frame_tests'
    compiled=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic',
        '-Wno-attributes','-Wno-unknown-pragmas','-Dlinux','-isystem',str(runtime/'include'),
        str(ROOT/'geomagic/tests/teleop_geo_frame_test.cpp'),'-L'+str(runtime/'lib'),
        '-lHDU','-lrt','-pthread','-o',str(binary)],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    dependencies=subprocess.run(['ldd',str(binary)],capture_output=True,text=True)
    assert 'libHD.so' not in dependencies.stdout and 'libPhantom' not in dependencies.stdout
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stdout+result.stderr
    assert '8 passed, 0 failed, 0 skipped' in result.stdout

