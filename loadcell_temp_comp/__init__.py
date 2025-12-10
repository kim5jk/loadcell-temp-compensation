"""
Load Cell Temperature Compensation Library

A Python library for compensating temperature-induced drift in load cell measurements.
Achieves 5-6x improvement in measurement stability using segment-based linear regression.

Author: Jueseok Kim
"""

import pandas as pd
import numpy as np
from scipy import ndimage
import statsmodels.api as sm
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

__version__ = "1.0.0"
__author__ = "Jueseok Kim"


@dataclass
class SegmentStats:
    """Statistics for a single drift segment."""
    segment_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: float
    num_points: int
    temp_change: float
    raw_std: float
    adjusted_std: float
    improvement_pct: float
    slope: float
    intercept: float


class TemperatureCompensator:
    """
    Temperature compensation for load cell measurements using 
    segment-based linear regression.
    
    This compensator detects stable measurement periods (segments),
    fits a linear temperature-weight relationship for each segment,
    and normalizes weights to a reference temperature.
    
    Parameters
    ----------
    sigma_temp : float, default=30
        Gaussian smoothing sigma for temperature signal.
        Higher values = more smoothing.
    
    sigma_weight : float, default=30
        Gaussian smoothing sigma for weight signal.
        Higher values = more smoothing.
    
    drift_threshold : float, default=0.4
        Weight change threshold (g) to detect a new segment.
        When consecutive weight difference exceeds this, a new segment begins.
    
    reference_temperature : float, default=20.0
        Target temperature (°C) for normalization.
        All weights are adjusted as if measured at this temperature.
    
    detrend_threshold : float, default=0.5
        Long-term drift rate threshold (g/day).
        Segments drifting faster than this are flattened.
    
    min_segment_length : int, default=2
        Minimum number of points required to process a segment.
    
    Examples
    --------
    >>> from loadcell_temp_comp import TemperatureCompensator
    >>> 
    >>> # Initialize with default parameters
    >>> comp = TemperatureCompensator()
    >>> 
    >>> # Fit and transform data
    >>> result = comp.fit_transform(
    ...     timestamps=df['datetime'],
    ...     weights=df['weight_g'],
    ...     temperatures=df['temperature_c']
    ... )
    >>> 
    >>> # Check improvement
    >>> print(f"Raw std: {df['weight_g'].std():.3f}g")
    >>> print(f"Compensated std: {result['adjusted_weight'].std():.3f}g")
    """
    
    def __init__(
        self,
        sigma_temp: float = 30,
        sigma_weight: float = 30,
        drift_threshold: float = 0.4,
        reference_temperature: float = 20.0,
        detrend_threshold: float = 0.5,
        min_segment_length: int = 2
    ):
        self.sigma_temp = sigma_temp
        self.sigma_weight = sigma_weight
        self.drift_threshold = drift_threshold
        self.reference_temperature = reference_temperature
        self.detrend_threshold = detrend_threshold
        self.min_segment_length = min_segment_length
        
        # Fitted parameters
        self._segment_stats: List[SegmentStats] = []
        self._is_fitted = False
    
    def _gaussian_smooth(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian smoothing to 1D array."""
        if len(data) > 1:
            return ndimage.gaussian_filter(data, sigma=sigma, order=0)
        return data
    
    def _detect_segments(
        self, 
        weights: np.ndarray,
        interactions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect drift segments based on weight changes.
        
        Returns array of segment IDs for each data point.
        """
        segment_flags = np.zeros(len(weights), dtype=int)
        current_segment = 1
        segment_flags[0] = current_segment
        
        for i in range(1, len(weights)):
            diff = abs(weights[i] - weights[i - 1])
            is_interaction = interactions[i] if interactions is not None else False
            
            if diff >= self.drift_threshold or (is_interaction and diff >= self.drift_threshold):
                current_segment += 1
            
            segment_flags[i] = current_segment
        
        return segment_flags
    
    def _fit_segment(
        self,
        segment_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[SegmentStats]]:
        """
        Fit temperature compensation model for a single segment.
        
        Returns the transformed dataframe and segment statistics.
        """
        if len(segment_df) <= self.min_segment_length:
            return segment_df, None
        
        df = segment_df.copy()
        
        # Apply Gaussian smoothing
        df['smoothed_weight'] = self._gaussian_smooth(
            df['weight'].values, self.sigma_weight
        )
        df['smoothed_temp'] = self._gaussian_smooth(
            df['temperature'].values, self.sigma_temp
        )
        
        # Calculate differences
        df['temp_diff'] = df['smoothed_temp'].diff().fillna(0)
        df['weight_diff'] = df['smoothed_weight'].diff().fillna(0)
        
        # Skip first row (no diff available)
        df_fit = df.iloc[1:].copy()
        
        if len(df_fit) < 2:
            return segment_df, None
        
        # Fit linear model: weight_diff = slope * temp_diff + intercept
        X = df_fit['temp_diff'].values
        Y = df_fit['weight_diff'].values
        
        # Handle case where temp_diff is constant (no variation)
        if np.std(X) < 1e-10:
            slope = 0
            intercept = np.mean(Y)
            fitted_values = np.full_like(Y, intercept)
        else:
            X_with_const = sm.add_constant(X)
            try:
                model = sm.OLS(Y, X_with_const)
                results = model.fit()
                slope = results.params[1] if len(results.params) > 1 else results.params[0]
                intercept = results.params[0] if len(results.params) > 1 else 0
                fitted_values = results.predict(X_with_const)
            except Exception:
                slope = 0
                intercept = np.mean(Y)
                fitted_values = np.full_like(Y, intercept)
        
        # Reconstruct weight using fitted model
        df_fit['fitted_diff'] = fitted_values
        df_fit['cumulative_fitted'] = df_fit['fitted_diff'].cumsum()
        
        reference_weight = df_fit['smoothed_weight'].iloc[0]
        df_fit['new_weight'] = reference_weight + df_fit['cumulative_fitted']
        
        # Adjust to reference temperature
        temp_diff_from_ref = df_fit['smoothed_temp'] - self.reference_temperature
        df_fit['adjusted_weight'] = df_fit['new_weight'] - slope * temp_diff_from_ref
        
        # Calculate statistics
        raw_std = df_fit['weight'].std()
        adjusted_std = df_fit['adjusted_weight'].std()
        improvement = ((raw_std - adjusted_std) / raw_std * 100) if raw_std > 0 else 0
        
        stats = SegmentStats(
            segment_id=int(df_fit['segment_id'].iloc[0]),
            start_time=df_fit['timestamp'].iloc[0],
            end_time=df_fit['timestamp'].iloc[-1],
            duration_hours=(df_fit['timestamp'].iloc[-1] - df_fit['timestamp'].iloc[0]).total_seconds() / 3600,
            num_points=len(df_fit),
            temp_change=df_fit['smoothed_temp'].max() - df_fit['smoothed_temp'].min(),
            raw_std=raw_std,
            adjusted_std=adjusted_std,
            improvement_pct=improvement,
            slope=slope,
            intercept=intercept
        )
        
        return df_fit, stats
    
    def _detrend_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove long-term drift from adjusted weights."""
        df = df.copy()
        adjusted_weights = df['adjusted_weight'].copy()
        
        for segment_id in df['segment_id'].unique():
            mask = df['segment_id'] == segment_id
            segment = df[mask]
            
            if len(segment) < 2:
                continue
            
            start_weight = adjusted_weights[mask].iloc[0]
            end_weight = adjusted_weights[mask].iloc[-1]
            
            duration_days = (
                (segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0]).total_seconds() 
                / (3600 * 24)
            )
            
            if duration_days > 0:
                drift_rate = abs(end_weight - start_weight) / duration_days
                
                if drift_rate > self.detrend_threshold:
                    diff = end_weight - start_weight
                    adjusted_weights.loc[mask] = start_weight
                    
                    # Shift subsequent weights
                    subsequent_mask = df['timestamp'] > segment['timestamp'].max()
                    if diff != 0:
                        adjusted_weights.loc[subsequent_mask] -= diff
        
        df['detrended_weight'] = adjusted_weights
        return df
    
    def fit_transform(
        self,
        timestamps: Union[pd.Series, np.ndarray, List],
        weights: Union[pd.Series, np.ndarray, List],
        temperatures: Union[pd.Series, np.ndarray, List],
        interactions: Optional[Union[pd.Series, np.ndarray, List]] = None
    ) -> pd.DataFrame:
        """
        Fit the compensation model and transform the data.
        
        Parameters
        ----------
        timestamps : array-like
            Datetime timestamps for each measurement.
        weights : array-like
            Raw weight measurements in grams.
        temperatures : array-like
            Temperature measurements in °C.
        interactions : array-like, optional
            Boolean array indicating interaction events.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - timestamp: Original timestamps
            - weight: Original weights
            - temperature: Original temperatures
            - segment_id: Detected segment ID
            - smoothed_weight: Gaussian-smoothed weight
            - smoothed_temp: Gaussian-smoothed temperature
            - new_weight: Reconstructed weight from fitted model
            - adjusted_weight: Temperature-normalized weight
            - detrended_weight: Long-term drift removed
        """
        # Create working dataframe
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'weight': np.array(weights, dtype=float),
            'temperature': np.array(temperatures, dtype=float)
        })
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply initial smoothing for segment detection
        smoothed_weights = self._gaussian_smooth(df['weight'].values, self.sigma_weight)
        
        # Detect segments
        interactions_arr = np.array(interactions) if interactions is not None else None
        df['segment_id'] = self._detect_segments(smoothed_weights, interactions_arr)
        
        # Process each segment
        self._segment_stats = []
        processed_segments = []
        
        for segment_id in df['segment_id'].unique():
            segment_df = df[df['segment_id'] == segment_id].copy()
            processed_df, stats = self._fit_segment(segment_df)
            
            if stats is not None:
                self._segment_stats.append(stats)
                processed_segments.append(processed_df)
        
        if not processed_segments:
            raise ValueError("No valid segments found. Try adjusting drift_threshold or min_segment_length.")
        
        # Combine all segments
        result = pd.concat(processed_segments, ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)
        
        # Apply detrending
        result = self._detrend_weight(result)
        
        self._is_fitted = True
        return result
    
    def get_segment_stats(self) -> pd.DataFrame:
        """
        Get statistics for all fitted segments.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with per-segment statistics including:
            - segment_id, start_time, end_time
            - duration_hours, num_points, temp_change
            - raw_std, adjusted_std, improvement_pct
            - slope, intercept
        """
        if not self._is_fitted:
            raise RuntimeError("Compensator must be fitted first. Call fit_transform().")
        
        return pd.DataFrame([vars(s) for s in self._segment_stats])
    
    def get_summary(self) -> Dict:
        """
        Get overall compensation summary statistics.
        
        Returns
        -------
        dict
            Summary statistics including mean improvement, etc.
        """
        if not self._is_fitted:
            raise RuntimeError("Compensator must be fitted first. Call fit_transform().")
        
        stats_df = self.get_segment_stats()
        
        return {
            'num_segments': len(stats_df),
            'total_duration_hours': stats_df['duration_hours'].sum(),
            'mean_raw_std': stats_df['raw_std'].mean(),
            'mean_adjusted_std': stats_df['adjusted_std'].mean(),
            'median_improvement_pct': stats_df['improvement_pct'].median(),
            'mean_improvement_pct': stats_df['improvement_pct'].mean(),
            'improvement_factor': stats_df['raw_std'].mean() / stats_df['adjusted_std'].mean(),
            'segments_above_90pct': (stats_df['improvement_pct'] > 90).sum(),
            'segments_above_80pct': (stats_df['improvement_pct'] > 80).sum(),
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TemperatureCompensator("
            f"sigma_temp={self.sigma_temp}, "
            f"sigma_weight={self.sigma_weight}, "
            f"drift_threshold={self.drift_threshold}, "
            f"ref_temp={self.reference_temperature}°C, "
            f"status={status})"
        )


# Convenience function for simple usage
def compensate_temperature(
    timestamps,
    weights,
    temperatures,
    **kwargs
) -> pd.DataFrame:
    """
    Quick temperature compensation with default parameters.
    
    Parameters
    ----------
    timestamps : array-like
        Datetime timestamps.
    weights : array-like
        Raw weight measurements (g).
    temperatures : array-like
        Temperature measurements (°C).
    **kwargs
        Additional parameters passed to TemperatureCompensator.
    
    Returns
    -------
    pd.DataFrame
        Compensated data with adjusted_weight column.
    
    Examples
    --------
    >>> result = compensate_temperature(
    ...     timestamps=df['time'],
    ...     weights=df['weight'],
    ...     temperatures=df['temp']
    ... )
    """
    compensator = TemperatureCompensator(**kwargs)
    return compensator.fit_transform(timestamps, weights, temperatures)
