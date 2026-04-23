"""Spatial subsetting helpers — SEVIRI lat/lon are 2-D, not dimension coords."""

from __future__ import annotations

import numpy as np
import xarray as xr


def bbox_subset(
    ds: xr.Dataset,
    lon: tuple[float, float],
    lat: tuple[float, float],
) -> xr.Dataset:
    """Return a sub-dataset restricted to the tightest bounding box that covers
    every pixel whose (lat, lon) falls inside the requested window.

    The SEVIRI grid is irregular in lat/lon, so we keep the full ``(along_track,
    across_track)`` structure and just clip to the minimal index window enclosing
    the selected pixels. A boolean ``in_bbox`` mask is added for downstream use.
    """
    lo_lon, hi_lon = sorted(lon)
    lo_lat, hi_lat = sorted(lat)
    in_bbox = (
        (ds["lon"] >= lo_lon) & (ds["lon"] <= hi_lon)
        & (ds["lat"] >= lo_lat) & (ds["lat"] <= hi_lat)
    )
    mask = in_bbox.compute() if hasattr(in_bbox, "compute") else in_bbox
    if not bool(mask.any()):
        empty = ds.isel(along_track=slice(0, 0), across_track=slice(0, 0))
        empty = empty.assign(in_bbox=("along_track", "across_track"),
                             in_bbox_=mask)
        return empty
    at_any = mask.any(dim="across_track").values
    ac_any = mask.any(dim="along_track").values
    at_idx = np.where(at_any)[0]
    ac_idx = np.where(ac_any)[0]
    sub = ds.isel(
        along_track=slice(int(at_idx[0]), int(at_idx[-1]) + 1),
        across_track=slice(int(ac_idx[0]), int(ac_idx[-1]) + 1),
    )
    sub = sub.assign(in_bbox=mask.isel(
        along_track=slice(int(at_idx[0]), int(at_idx[-1]) + 1),
        across_track=slice(int(ac_idx[0]), int(ac_idx[-1]) + 1),
    ))
    return sub


def nearest_pixel(
    ds: xr.Dataset,
    lon: float,
    lat: float,
    max_distance_km: float | None = None,
) -> xr.Dataset:
    """Return a single-pixel dataset at the nearest on-disk pixel.

    Distance is great-circle (haversine). If ``max_distance_km`` is set and the
    nearest pixel is further, raises ``ValueError``.
    """
    lon_a = np.asarray(ds["lon"].values, dtype=np.float64)
    lat_a = np.asarray(ds["lat"].values, dtype=np.float64)
    valid = np.isfinite(lon_a) & np.isfinite(lat_a)
    if not valid.any():
        raise ValueError("No on-disk pixels in dataset")

    phi1 = np.deg2rad(lat)
    phi2 = np.deg2rad(lat_a)
    dphi = phi2 - phi1
    dlam = np.deg2rad(lon_a - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    d_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    d_km = np.where(valid, d_km, np.inf)

    ij = np.unravel_index(np.argmin(d_km), d_km.shape)
    dist = float(d_km[ij])
    if max_distance_km is not None and dist > max_distance_km:
        raise ValueError(
            f"Nearest pixel is {dist:.1f} km away (max allowed {max_distance_km})"
        )
    point = ds.isel(along_track=int(ij[0]), across_track=int(ij[1]))
    point.attrs["distance_km"] = dist
    point.attrs["requested_lon"] = lon
    point.attrs["requested_lat"] = lat
    return point
