"""Unit tests for compare_1d.py.

They use small generated files, so they run in seconds. Run them with
``python -m pytest .github/scripts/tests`` or
``python -m unittest discover -s .github/scripts/tests``.
"""

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'compare_1d.py'
_spec = importlib.util.spec_from_file_location('compare_1d', SCRIPT)
compare_1d = importlib.util.module_from_spec(_spec)
sys.modules['compare_1d'] = compare_1d
_spec.loader.exec_module(compare_1d)

ENV = 'role={role}\nocean_bgc_sha={sha}\ncefi_sha=abc123\ngfortran=GNU 13.2\n'


def write_nc(path, variables, fmt='NETCDF4_CLASSIC', global_attrs=None, var_attrs=None):
    """Write a small netCDF file; dimensions are named after their sizes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, 'w', format=fmt) as ds:
        for key, value in (global_attrs or {}).items():
            ds.setncattr(key, value)
        for name, values in variables.items():
            data = np.asarray(values)
            dims = []
            for size in data.shape:
                dim = f'n{size}'
                if dim not in ds.dimensions:
                    ds.createDimension(dim, size)
                dims.append(dim)
            var = ds.createVariable(name, data.dtype, tuple(dims))
            for key, value in (var_attrs or {}).get(name, {}).items():
                var.setncattr(key, value)
            var[...] = data


def write_text(path, text):
    """Write a small text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


class CompareTestCase(unittest.TestCase):
    """Base class: a base and a PR folder in a temporary directory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name) / 'base'
        self.pr = Path(tmp.name) / 'pr'
        self.base.mkdir()
        self.pr.mkdir()
        write_text(self.base / 'env.txt', ENV.format(role='base', sha='aaa'))
        write_text(self.pr / 'env.txt', ENV.format(role='pr', sha='bbb'))

    def both(self, rel, variables, **kwargs):
        """Write the same netCDF file on both sides."""
        write_nc(self.base / rel, variables, **kwargs)
        write_nc(self.pr / rel, variables, **kwargs)

    def compare(self):
        return compare_1d.compare_dirs(self.base, self.pr)


class TestIdentical(CompareTestCase):
    def test_identical_outputs(self):
        temp = np.linspace(1.0, 2.0, 12).reshape(3, 4)
        self.both('RESTART_48hrs/MOM.res.nc', {'Temp': temp, 'dic': temp * 2})
        self.both('20040101.ocean_daily.nc', {'chl': temp[0]})
        for side in (self.base, self.pr):
            write_text(side / 'RESTART_48hrs/ocean.stats', 'step 1 En 1.0\n')
        report = self.compare()
        self.assertFalse(report.answer_change)
        self.assertTrue(report.valid)
        self.assertEqual(report.files_compared, 3)
        self.assertEqual(report.vars_compared, 3)
        self.assertIn('No answer change', compare_1d.headline(report))

    def test_nan_in_same_place_is_equal(self):
        self.both('a.nc', {'o2': np.array([1.0, np.nan, 3.0])})
        self.assertFalse(self.compare().answer_change)

    def test_classic_and_netcdf4_formats_with_same_data(self):
        data = {'alk': np.array([2300.5, 2310.25]), 'nlev': np.arange(2, dtype='i4')}
        write_nc(self.base / 'a.nc', data, fmt='NETCDF3_64BIT_OFFSET')
        write_nc(self.pr / 'a.nc', data, fmt='NETCDF4')
        report = self.compare()
        self.assertFalse(report.answer_change, report.var_diffs)


class TestAnswerChanges(CompareTestCase):
    def test_one_changed_value(self):
        base = np.zeros((2, 3))
        pr = base.copy()
        pr[1, 2] = 1.0e-12
        write_nc(self.base / 'RESTART_48hrs/MOM.res.nc', {'dic': base})
        write_nc(self.pr / 'RESTART_48hrs/MOM.res.nc', {'dic': pr})
        report = self.compare()
        self.assertTrue(report.answer_change)
        [diff] = report.var_diffs
        found = (diff.name, diff.reason, diff.n_diff, diff.n_total)
        self.assertEqual(found, ('dic', 'data', 1, 6))
        self.assertEqual(diff.first_index, [1, 2])
        self.assertAlmostEqual(diff.max_abs_diff, 1.0e-12)
        self.assertIn('Answer change', compare_1d.headline(report))

    def test_last_bit_change_is_found(self):
        value = np.array([0.1])
        write_nc(self.base / 'a.nc', {'po4': value})
        write_nc(self.pr / 'a.nc', {'po4': np.nextafter(value, 1.0)})
        self.assertTrue(self.compare().answer_change)

    def test_negative_zero_differs_from_zero(self):
        write_nc(self.base / 'a.nc', {'fed': np.array([0.0])})
        write_nc(self.pr / 'a.nc', {'fed': np.array([-0.0])})
        self.assertTrue(self.compare().answer_change)

    def test_nan_versus_number_differs(self):
        write_nc(self.base / 'a.nc', {'o2': np.array([1.0, 2.0])})
        write_nc(self.pr / 'a.nc', {'o2': np.array([1.0, np.nan])})
        report = self.compare()
        [diff] = report.var_diffs
        self.assertEqual(diff.n_diff, 1)
        self.assertIsNone(diff.max_abs_diff)

    def test_dtype_change(self):
        write_nc(self.base / 'a.nc', {'nh4': np.array([1.0], dtype='f8')})
        write_nc(self.pr / 'a.nc', {'nh4': np.array([1.0], dtype='f4')})
        [diff] = self.compare().var_diffs
        self.assertEqual(diff.reason, 'dtype')

    def test_shape_change(self):
        write_nc(self.base / 'a.nc', {'no3': np.zeros(3)})
        write_nc(self.pr / 'a.nc', {'no3': np.zeros(4)})
        [diff] = self.compare().var_diffs
        self.assertEqual(diff.reason, 'shape')

    def test_text_output_change(self):
        write_text(self.base / 'RESTART_48hrs/ocean.stats', 'En 1.0\n')
        write_text(self.pr / 'RESTART_48hrs/ocean.stats', 'En 1.1\n')
        report = self.compare()
        self.assertTrue(report.answer_change)
        [diff] = report.text_diffs
        self.assertIn('+En 1.1', diff.lines)

    def test_removed_restart_variable_is_an_answer_change(self):
        rel = 'RESTART_48hrs/MOM.res.nc'
        write_nc(self.base / rel, {'Temp': [1.0], 'co3_ion': [2.0]})
        write_nc(self.pr / rel, {'Temp': [1.0]})
        report = self.compare()
        self.assertEqual(report.removed_restart, ['RESTART_48hrs/MOM.res.nc: co3_ion'])
        self.assertTrue(report.answer_change)

    def test_removed_restart_file_is_an_answer_change(self):
        self.both('common.nc', {'x': [0.0]})
        write_nc(self.base / 'RESTART_48hrs/ice_cobalt.res.nc', {'x': [1.0]})
        report = self.compare()
        self.assertEqual(report.removed_restart, ['RESTART_48hrs/ice_cobalt.res.nc'])
        self.assertTrue(report.answer_change)


class TestOtherChanges(CompareTestCase):
    def test_added_diag_variable_is_not_an_answer_change(self):
        rel = '20040101.ocean_cobalt.nc'
        write_nc(self.base / rel, {'chl': [1.0]})
        write_nc(self.pr / rel, {'chl': [1.0], 'new_diag': [3.0]})
        report = self.compare()
        self.assertFalse(report.answer_change)
        self.assertEqual(report.added, ['20040101.ocean_cobalt.nc: new_diag'])

    def test_removed_diag_file_is_not_an_answer_change(self):
        self.both('common.nc', {'x': [0.0]})
        write_nc(self.base / '20040101.ocean_extra.nc', {'chl': [1.0]})
        report = self.compare()
        self.assertFalse(report.answer_change)
        self.assertEqual(report.removed, ['20040101.ocean_extra.nc'])

    def test_added_file(self):
        self.both('common.nc', {'x': [0.0]})
        write_nc(self.pr / '20040101.new.nc', {'chl': [1.0]})
        self.assertEqual(self.compare().added, ['20040101.new.nc'])

    def test_parameter_doc_change_is_not_an_answer_change(self):
        write_text(self.base / 'COBALT_parameter_doc.all', 'A = 1\n')
        write_text(self.pr / 'COBALT_parameter_doc.all', 'A = 1\nNEW_OPTION = False\n')
        write_text(self.base / 'available_diags.000000', 'chl\n')
        write_text(self.pr / 'available_diags.000000', 'chl\n')
        report = self.compare()
        self.assertFalse(report.answer_change)
        [diff] = report.param_diffs
        self.assertEqual(diff.file, 'COBALT_parameter_doc.all')
        self.assertIn('+NEW_OPTION = False', diff.lines)

    def test_attribute_only_change_is_metadata(self):
        write_nc(self.base / 'a.nc', {'chl': [1.0]}, var_attrs={'chl': {'units': 'mg'}})
        write_nc(self.pr / 'a.nc', {'chl': [1.0]}, var_attrs={'chl': {'units': 'kg'}})
        report = self.compare()
        self.assertFalse(report.answer_change)
        self.assertEqual(report.metadata, ['a.nc: chl: attribute units changed'])

    def test_checksum_attribute_not_repeated_for_changed_data(self):
        attrs_a, attrs_b = {'t': {'checksum': 'A'}}, {'t': {'checksum': 'B'}}
        write_nc(self.base / 'a.res.nc', {'t': [1.0]}, var_attrs=attrs_a)
        write_nc(self.pr / 'a.res.nc', {'t': [2.0]}, var_attrs=attrs_b)
        report = self.compare()
        self.assertEqual(len(report.var_diffs), 1)
        self.assertEqual(report.metadata, [])

    def test_distributed_file_is_read_as_netcdf(self):
        write_nc(self.base / '20040102.ocean_daily_subset.nc.0000', {'chl': [1.0]})
        write_nc(self.pr / '20040102.ocean_daily_subset.nc.0000', {'chl': [2.0]})
        [diff] = self.compare().var_diffs
        self.assertEqual(diff.reason, 'data')


class TestEnvironment(CompareTestCase):
    def test_different_setup_is_not_valid(self):
        self.both('common.nc', {'x': [0.0]})
        env = ENV.format(role='pr', sha='bbb').replace('abc123', 'zzz')
        write_text(self.pr / 'env.txt', env)
        report = self.compare()
        self.assertFalse(report.valid)
        self.assertEqual(report.env_mismatch, {'cefi_sha': ['abc123', 'zzz']})
        self.assertIn('not valid', compare_1d.headline(report))

    def test_different_cpu_is_info_only(self):
        base_env = ENV.format(role='base', sha='a') + 'cpu_model=EPYC 7763\n'
        write_text(self.base / 'env.txt', base_env)
        pr_env = ENV.format(role='pr', sha='b') + 'cpu_model=EPYC 9V74\n'
        write_text(self.pr / 'env.txt', pr_env)
        write_nc(self.base / 'a.nc', {'o2': [1.0]})
        write_nc(self.pr / 'a.nc', {'o2': [1.5]})
        report = self.compare()
        self.assertTrue(report.valid)
        self.assertIn('cpu_model', report.env_info)
        self.assertIn('possibly the environment', compare_1d.headline(report))

    def test_missing_env_is_reported(self):
        self.both('common.nc', {'x': [0.0]})
        (self.base / 'env.txt').unlink()
        report = self.compare()
        self.assertEqual(report.env_missing, ['base'])
        self.assertTrue(report.valid)

    def test_folder_with_only_env_counts_as_missing(self):
        write_nc(self.base / 'a.nc', {'o2': [1.0]})
        report = self.compare()
        self.assertEqual(report.missing, ['PR'])
        self.assertFalse(report.answer_change)

    def test_missing_output_folder(self):
        missing = self.pr.parent / 'not_there'
        report = compare_1d.compare_dirs(self.base, missing)
        self.assertEqual(report.missing, ['PR'])
        self.assertIn('No comparison', compare_1d.headline(report))


class TestCommandLine(CompareTestCase):
    def run_main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = compare_1d.main([str(self.base), str(self.pr), *args])
        return status, out.getvalue()

    def make_answer_change(self):
        write_nc(self.base / 'RESTART_48hrs/MOM.res.nc', {'dic': [1.0]})
        write_nc(self.pr / 'RESTART_48hrs/MOM.res.nc', {'dic': [2.0]})

    def test_report_only_exit_status_is_zero(self):
        self.make_answer_change()
        status, out = self.run_main()
        self.assertEqual(status, 0)
        self.assertIn('| 1. Answers | CHANGED: 1 variable with different data |', out)

    def test_fail_on_diff(self):
        self.make_answer_change()
        self.assertEqual(self.run_main('--fail-on-diff')[0], 1)

    def test_fail_on_diff_without_change(self):
        self.both('a.nc', {'dic': [1.0]})
        self.assertEqual(self.run_main('--fail-on-diff')[0], 0)

    def test_fail_on_diff_with_missing_output(self):
        args = [str(self.base), str(self.pr / 'none'), '--fail-on-diff']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare_1d.main(args), 2)

    def test_summary_json_and_annotation(self):
        self.make_answer_change()
        summary = self.base.parent / 'summary.md'
        json_path = self.base.parent / 'report.json'
        status, out = self.run_main(
            '--summary', str(summary), '--json', str(json_path), '--github-annotations'
        )
        self.assertEqual(status, 0)
        text = summary.read_text(encoding='utf-8')
        rows = (
            '1. Answers',
            '2. Outputs added or removed',
            '3. Parameters / diag list',
            '4. Metadata only',
            'Environment',
        )
        for row in rows:
            self.assertIn(row, text)
        self.assertIn('::warning title=1D answer report::Answer change', out)
        self.assertIn('"answer_change": true', json_path.read_text(encoding='utf-8'))

    def test_no_annotation_without_change(self):
        self.both('a.nc', {'dic': [1.0]})
        _, out = self.run_main('--github-annotations')
        self.assertNotIn('::warning', out)


class TestCollectOutputs(unittest.TestCase):
    """collect_1d_outputs.sh decides what gets compared, so test what it copies."""

    collect = SCRIPT.parent / 'collect_1d_outputs.sh'

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run_dir = self.root / 'CEFI' / 'exps' / 'OM4.single_column.COBALT'
        outputs = [
            'RESTART_48hrs/MOM.res.nc',
            'RESTART_48hrs/ocean.stats',
            'RESTART_24hrs/ocean.stats',
            '20040101.ocean_cobalt_budget.nc',
            '20040102.ocean_daily_subset.nc.0000',
            'MOM_parameter_doc.all',
            'COBALT_parameter_doc.short',
            'available_diags.000000',
            'SIS.available_diags',
        ]
        not_outputs = ['o2_col_budget.nc', 'ncap2_tmp.nc', 'out1', 'input.nml']
        for rel in outputs + not_outputs:
            write_text(self.run_dir / rel, rel)
        self.expected = {*outputs, 'env.txt'}

    def run_collect(self, dest):
        return subprocess.run(
            ['bash', str(self.collect), str(dest), 'pr', 'image:tag'],
            cwd=self.run_dir,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_copies_outputs_only(self):
        dest = self.root / 'out'
        proc = self.run_collect(dest)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        copied = set(compare_1d.list_files(dest))
        self.assertEqual(copied, self.expected)

    def test_env_has_every_key(self):
        dest = self.root / 'out'
        self.run_collect(dest)
        env = compare_1d.read_env(dest / 'env.txt')
        keys = {
            'role', 'image', 'ocean_bgc_sha', 'cefi_sha', 'gfortran', 'netcdf_c',
            'netcdf_fortran', 'mpi', 'glibc', 'cpu_model',
        }
        self.assertTrue(keys <= set(env), keys - set(env))
        self.assertEqual((env['role'], env['image']), ('pr', 'image:tag'))

    def test_existing_destination_is_refused(self):
        dest = self.root / 'out'
        dest.mkdir()
        self.assertEqual(self.run_collect(dest).returncode, 2)


if __name__ == '__main__':
    unittest.main()
