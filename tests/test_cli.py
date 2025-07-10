"""Tests for the command-line interface."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from lorenz_attractor.cli import cmd_simulate, create_parser


class TestCLIParser:
    """Test suite for CLI argument parser."""
    
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description == 'Professional Lorenz Attractor Simulation Suite'
    
    def test_simulate_command_default_args(self):
        """Test simulate command with default arguments."""
        parser = create_parser()
        args = parser.parse_args(['simulate'])
        
        assert args.command == 'simulate'
        assert args.sigma == 10.0
        assert args.rho == 28.0
        assert args.beta == 8/3
        assert args.x0 == 1.0
        assert args.y0 == 1.0
        assert args.z0 == 1.0
        assert args.dt == 0.01
        assert args.steps == 10000
        assert args.method == 'rk4'
        assert not args.plot
        assert not args.random_ic
        assert not args.true_random
    
    def test_simulate_command_custom_args(self):
        """Test simulate command with custom arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'simulate', 
            '--sigma', '5.0',
            '--rho', '15.0',
            '--beta', '2.0',
            '--x0', '2.0',
            '--y0', '3.0',
            '--z0', '4.0',
            '--dt', '0.001',
            '--steps', '50000',
            '--method', 'euler',
            '--plot',
            '--random-ic'
        ])
        
        assert args.sigma == 5.0
        assert args.rho == 15.0
        assert args.beta == 2.0
        assert args.x0 == 2.0
        assert args.y0 == 3.0
        assert args.z0 == 4.0
        assert args.dt == 0.001
        assert args.steps == 50000
        assert args.method == 'euler'
        assert args.plot
        assert args.random_ic
    
    def test_realtime_command_default_args(self):
        """Test realtime command with default arguments."""
        parser = create_parser()
        args = parser.parse_args(['realtime'])
        
        assert args.command == 'realtime'
        assert args.method == 'matplotlib'
        assert args.sigma == 10.0
        assert args.rho == 28.0
        assert args.beta == 8/3
        assert args.dt == 0.01
        assert args.trail == 2000
    
    def test_sweep_command_required_args(self):
        """Test sweep command with required arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'sweep',
            '--parameter', 'rho',
            '--range', '20', '30'
        ])
        
        assert args.command == 'sweep'
        assert args.parameter == 'rho'
        assert args.range == [20.0, 30.0]
        assert args.steps == 50
    
    def test_bifurcation_command_required_args(self):
        """Test bifurcation command with required arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'bifurcation',
            '--parameter', 'rho',
            '--range', '0', '50'
        ])
        
        assert args.command == 'bifurcation'
        assert args.parameter == 'rho'
        assert args.range == [0.0, 50.0]
        assert args.points == 200
    
    def test_analysis_command_required_args(self):
        """Test analysis command with required arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'analysis',
            '--type', 'lyapunov'
        ])
        
        assert args.command == 'analysis'
        assert args.type == 'lyapunov'
        assert args.sigma == 10.0
        assert args.steps == 50000
    
    def test_web_command_default_args(self):
        """Test web command with default arguments."""
        parser = create_parser()
        args = parser.parse_args(['web'])
        
        assert args.command == 'web'
        assert args.host == 'localhost'
        assert args.port == 8050
        assert not args.debug
    
    def test_invalid_method_choice(self):
        """Test that invalid method choices raise an error."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['simulate', '--method', 'invalid'])
    
    def test_invalid_parameter_choice(self):
        """Test that invalid parameter choices raise an error."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['sweep', '--parameter', 'invalid', '--range', '0', '1'])
    
    def test_missing_required_args(self):
        """Test that missing required arguments raise an error."""
        parser = create_parser()
        
        # sweep command requires --parameter and --range
        with pytest.raises(SystemExit):
            parser.parse_args(['sweep'])
        
        # bifurcation command requires --parameter and --range
        with pytest.raises(SystemExit):
            parser.parse_args(['bifurcation'])
        
        # analysis command requires --type
        with pytest.raises(SystemExit):
            parser.parse_args(['analysis'])


class TestCLICommands:
    """Test suite for CLI command functions."""
    
    @patch('lorenz_attractor.cli.LorenzSystem')
    @patch('lorenz_attractor.cli.Simulator')
    @patch('builtins.print')
    def test_cmd_simulate_basic(self, mock_print, mock_simulator_class, mock_lorenz_class):
        """Test basic simulate command execution."""
        # Setup mocks
        mock_system = MagicMock()
        mock_lorenz_class.return_value = mock_system
        
        mock_simulator = MagicMock()
        mock_simulator_class.return_value = mock_simulator
        
        mock_result = MagicMock()
        mock_result.metadata = {'simulation_time': 1.23}
        mock_simulator.simulate.return_value = mock_result
        
        # Create mock args
        mock_args = MagicMock()
        mock_args.sigma = 10.0
        mock_args.rho = 28.0
        mock_args.beta = 8.0/3.0
        mock_args.x0 = 1.0
        mock_args.y0 = 1.0
        mock_args.z0 = 1.0
        mock_args.dt = 0.01
        mock_args.steps = 1000
        mock_args.method = 'rk4'
        mock_args.random_ic = False
        mock_args.true_random = False
        mock_args.output = None
        mock_args.export_data = None
        mock_args.plot = False
        mock_args.export_video = None
        
        # Execute command
        cmd_simulate(mock_args)
        
        # Verify system was created with correct parameters
        mock_lorenz_class.assert_called_once()
        
        # Verify simulator was created and simulate was called
        mock_simulator_class.assert_called_once_with(mock_system)
        mock_simulator.simulate.assert_called_once()
        
        # Verify print statements
        assert mock_print.call_count >= 3  # At least 3 print statements
    
    @patch('lorenz_attractor.cli.LorenzSystem')
    @patch('lorenz_attractor.cli.Simulator')
    @patch('lorenz_attractor.cli.InitialConditions')
    @patch('builtins.print')
    def test_cmd_simulate_random_ic(self, mock_print, mock_ic_class, mock_simulator_class, mock_lorenz_class):
        """Test simulate command with random initial conditions."""
        # Setup mocks
        mock_system = MagicMock()
        mock_lorenz_class.return_value = mock_system
        
        mock_simulator = MagicMock()
        mock_simulator_class.return_value = mock_simulator
        
        mock_result = MagicMock()
        mock_result.metadata = {'simulation_time': 1.23}
        mock_simulator.simulate.return_value = mock_result
        
        mock_random_ic = MagicMock()
        mock_random_ic.x = 1.0
        mock_random_ic.y = 2.0
        mock_random_ic.z = 3.0
        mock_ic_class.random.return_value = mock_random_ic
        
        # Create mock args
        mock_args = MagicMock()
        mock_args.sigma = 10.0
        mock_args.rho = 28.0
        mock_args.beta = 8.0/3.0
        mock_args.dt = 0.01
        mock_args.steps = 1000
        mock_args.method = 'rk4'
        mock_args.random_ic = True
        mock_args.true_random = False
        mock_args.output = None
        mock_args.export_data = None
        mock_args.plot = False
        mock_args.export_video = None
        
        # Execute command
        cmd_simulate(mock_args)
        
        # Verify random initial conditions were used
        mock_ic_class.random.assert_called_once_with(scale=2.0)
        mock_simulator.simulate.assert_called_once()
    
    @patch('lorenz_attractor.cli.DataExporter')
    @patch('lorenz_attractor.cli.LorenzSystem')
    @patch('lorenz_attractor.cli.Simulator')
    @patch('builtins.print')
    def test_cmd_simulate_with_data_export(self, mock_print, mock_simulator_class, mock_lorenz_class, mock_exporter_class):
        """Test simulate command with data export."""
        # Setup mocks
        mock_system = MagicMock()
        mock_lorenz_class.return_value = mock_system
        
        mock_simulator = MagicMock()
        mock_simulator_class.return_value = mock_simulator
        
        mock_result = MagicMock()
        mock_result.metadata = {'simulation_time': 1.23}
        mock_simulator.simulate.return_value = mock_result
        
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        
        # Create mock args
        mock_args = MagicMock()
        mock_args.sigma = 10.0
        mock_args.rho = 28.0
        mock_args.beta = 8.0/3.0
        mock_args.x0 = 1.0
        mock_args.y0 = 1.0
        mock_args.z0 = 1.0
        mock_args.dt = 0.01
        mock_args.steps = 1000
        mock_args.method = 'rk4'
        mock_args.random_ic = False
        mock_args.true_random = False
        mock_args.output = None
        mock_args.export_data = 'test_output.csv'
        mock_args.plot = False
        mock_args.export_video = None
        
        # Execute command
        cmd_simulate(mock_args)
        
        # Verify data export was called
        mock_exporter_class.assert_called_once()
        mock_exporter.export_csv.assert_called_once_with(mock_result, 'test_output.csv')