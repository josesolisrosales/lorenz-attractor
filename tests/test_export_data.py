"""Round-trip and format tests for DataExporter."""

import json

import numpy as np
import pytest

from lorenz_attractor.core.parameters import InitialConditions, SimulationConfig
from lorenz_attractor.core.simulator import Simulator
from lorenz_attractor.export.data import HAS_HDF5, DataExporter


@pytest.fixture
def result():
    sim = Simulator()
    return sim.simulate(
        InitialConditions(1.0, 1.0, 1.0), SimulationConfig(dt=0.01, num_steps=100)
    )


def test_export_csv_without_metadata_is_readable(result, tmp_path):
    path = tmp_path / 'out.csv'
    DataExporter().export_csv(result, str(path), include_metadata=False)
    text = path.read_text()
    assert text.splitlines()[0] == 'time,x,y,z'
    assert len(text.splitlines()) == 1 + len(result.time)


def test_export_json_roundtrips_trajectory(result, tmp_path):
    path = tmp_path / 'out.json'
    DataExporter().export_json(result, str(path))
    data = json.loads(path.read_text())
    assert data['parameters']['sigma'] == result.parameters.sigma
    traj = np.array(data['data']['trajectory'])
    np.testing.assert_allclose(traj, result.trajectory)


def test_export_numpy_roundtrips_via_simulationresult_load(result, tmp_path):
    from lorenz_attractor.core.simulator import SimulationResult

    path = tmp_path / 'out.npz'
    DataExporter().export_numpy(result, str(path))
    loaded = SimulationResult.load(str(path))
    np.testing.assert_allclose(loaded.trajectory, result.trajectory)


def test_export_matlab_writes_file(result, tmp_path):
    from scipy.io import loadmat

    path = tmp_path / 'out.mat'
    DataExporter().export_matlab(result, str(path))
    mat = loadmat(str(path))
    np.testing.assert_allclose(mat['trajectory'], result.trajectory)


@pytest.mark.skipif(not HAS_HDF5, reason='h5py not installed')
def test_export_hdf5_roundtrips_trajectory(result, tmp_path):
    import h5py

    path = tmp_path / 'out.h5'
    DataExporter().export_hdf5(result, str(path))
    with h5py.File(str(path), 'r') as f:
        assert 'data' in f
        assert 'trajectory' in f['data']
        np.testing.assert_allclose(f['data']['trajectory'][...], result.trajectory)
