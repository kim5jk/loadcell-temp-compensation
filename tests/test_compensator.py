"""
Tests for loadcell-temp-compensation package.
"""

import pytest
import numpy as np
import pandas as pd
from loadcell_temp_comp import TemperatureCompensator, compensate_temperature


@pytest.fixture
def sample_data():
    """Generate sample test data with known temperature drift."""
    np.random.seed(42)
    n_points = 200
    hours = np.linspace(0, 24, n_points)
    
    # Temperature with daily cycle
    temperature = 18 + 3 * np.sin(2 * np.pi * hours / 24)
    
    # Weight with 2 g/°C drift
    true_weight = 350
    temp_drift = 2.0 * (temperature - 18)
    noise = np.random.normal(0, 0.1, n_points)
    weight = true_weight + temp_drift + noise
    
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n_points, freq='7min'),
        'weight_g': weight,
        'temperature_c': temperature
    })
    return df


class TestTemperatureCompensator:
    """Test suite for TemperatureCompensator class."""
    
    def test_initialization(self):
        """Test compensator initializes with correct defaults."""
        comp = TemperatureCompensator()
        assert comp.sigma_temp == 30
        assert comp.sigma_weight == 30
        assert comp.drift_threshold == 0.4
        assert comp.reference_temperature == 20.0
        assert not comp._is_fitted
    
    def test_custom_parameters(self):
        """Test compensator accepts custom parameters."""
        comp = TemperatureCompensator(
            sigma_temp=50,
            sigma_weight=25,
            drift_threshold=0.5,
            reference_temperature=22.0
        )
        assert comp.sigma_temp == 50
        assert comp.sigma_weight == 25
        assert comp.drift_threshold == 0.5
        assert comp.reference_temperature == 22.0
    
    def test_fit_transform_returns_dataframe(self, sample_data):
        """Test fit_transform returns proper DataFrame."""
        comp = TemperatureCompensator()
        result = comp.fit_transform(
            timestamps=sample_data['datetime'],
            weights=sample_data['weight_g'],
            temperatures=sample_data['temperature_c']
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'adjusted_weight' in result.columns
        assert 'smoothed_temp' in result.columns
        assert 'segment_id' in result.columns
        assert comp._is_fitted
    
    def test_compensation_reduces_variance(self, sample_data):
        """Test that compensation reduces weight variance."""
        comp = TemperatureCompensator()
        result = comp.fit_transform(
            timestamps=sample_data['datetime'],
            weights=sample_data['weight_g'],
            temperatures=sample_data['temperature_c']
        )
        
        raw_std = sample_data['weight_g'].std()
        compensated_std = result['adjusted_weight'].std()
        
        # Compensated should have lower variance
        assert compensated_std < raw_std
    
    def test_get_segment_stats(self, sample_data):
        """Test segment statistics are returned correctly."""
        comp = TemperatureCompensator()
        comp.fit_transform(
            timestamps=sample_data['datetime'],
            weights=sample_data['weight_g'],
            temperatures=sample_data['temperature_c']
        )
        
        stats = comp.get_segment_stats()
        assert isinstance(stats, pd.DataFrame)
        assert 'segment_id' in stats.columns
        assert 'improvement_pct' in stats.columns
        assert 'slope' in stats.columns
    
    def test_get_summary(self, sample_data):
        """Test summary statistics are returned correctly."""
        comp = TemperatureCompensator()
        comp.fit_transform(
            timestamps=sample_data['datetime'],
            weights=sample_data['weight_g'],
            temperatures=sample_data['temperature_c']
        )
        
        summary = comp.get_summary()
        assert isinstance(summary, dict)
        assert 'num_segments' in summary
        assert 'mean_improvement_pct' in summary
        assert 'improvement_factor' in summary
    
    def test_get_stats_before_fit_raises(self):
        """Test accessing stats before fitting raises error."""
        comp = TemperatureCompensator()
        
        with pytest.raises(RuntimeError):
            comp.get_segment_stats()
        
        with pytest.raises(RuntimeError):
            comp.get_summary()


class TestConvenienceFunction:
    """Test suite for compensate_temperature function."""
    
    def test_basic_usage(self, sample_data):
        """Test convenience function works."""
        result = compensate_temperature(
            timestamps=sample_data['datetime'],
            weights=sample_data['weight_g'],
            temperatures=sample_data['temperature_c']
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'adjusted_weight' in result.columns
    
    def test_with_custom_params(self, sample_data):
        """Test convenience function accepts custom params."""
        result = compensate_temperature(
            timestamps=sample_data['datetime'],
            weights=sample_data['weight_g'],
            temperatures=sample_data['temperature_c'],
            sigma_temp=50,
            reference_temperature=22.0
        )
        
        assert isinstance(result, pd.DataFrame)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Test handling of empty data."""
        comp = TemperatureCompensator()
        
        with pytest.raises(Exception):
            comp.fit_transform(
                timestamps=[],
                weights=[],
                temperatures=[]
            )
    
    def test_constant_temperature(self):
        """Test handling of constant temperature."""
        n = 50
        comp = TemperatureCompensator(drift_threshold=10)  # High threshold to get single segment
        
        result = comp.fit_transform(
            timestamps=pd.date_range('2024-01-01', periods=n, freq='5min'),
            weights=np.full(n, 100.0) + np.random.normal(0, 0.1, n),
            temperatures=np.full(n, 20.0)
        )
        
        # Should still return valid result
        assert 'adjusted_weight' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
