import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pyproj import CRS, Transformer
import logging
import os
import csv
import math
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from osgeo import gdal  # Add this import at the top

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_1d_data_to_csv(data, output_path, attributes=None):
    """Save 1D dataset to CSV with metadata"""
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if attributes:
                for key, value in attributes.items():
                    writer.writerow([f"# {key}: {value}"])
            for value in data:
                writer.writerow([value])
        return True
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
        return False

def apply_lut_transformation(data, lut):
    """Apply Look-Up Table transformation to the data"""
    if lut is None:
        return data
    return np.interp(data, np.arange(len(lut)), lut)

def convert_to_cog(input_tiff, output_cog):
    """Convert a GeoTIFF to Cloud Optimized GeoTIFF"""
    try:
        with rasterio.open(input_tiff) as src:
            profile = src.profile.copy()
            
            # Update profile for COG
            profile.update({
                'driver': 'GTiff',
                'tiled': True,
                'compress': 'LZW',
                'blockxsize': 256,
                'blockysize': 256,
                'interleave': 'pixel',
            })

            # Calculate overview levels
            factors = []
            for i in range(1, 6):  # Create up to 5 overview levels
                factor = 2 ** i
                if (src.height // factor) > 50 and (src.width // factor) > 50:
                    factors.append(factor)

            with rasterio.open(output_cog, 'w', **profile) as dst:
                # Copy the data
                dst.write(src.read())
                
                # Build overviews
                dst.build_overviews(factors, Resampling.average)
                
                # Add overview statistics
                dst.update_tags(ns='rio_overview', resampling='average')
        
        # Remove the temporary file
        if os.path.exists(input_tiff):
            os.remove(input_tiff)
            
        return True
    except Exception as e:
        logger.error(f"Error converting to COG: {str(e)}")
        return False

def reproject_to_epsg3857(input_tif: str, output_tif: str) -> None:
    """Reproject a GeoTIFF to EPSG:3857"""
    warp_options = gdal.WarpOptions(
        dstSRS='EPSG:3857',
        format='GTiff',
        creationOptions=[
            'COMPRESS=LZW',
            'TILED=YES',
            'COPY_SRC_OVERVIEWS=YES',
            'BIGTIFF=YES',
            'RESAMPLING=NEAREST',
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512'
        ]
    )
    gdal.Warp(output_tif, input_tif, options=warp_options)
    os.remove(input_tif)  # Remove original TIFF after reprojection

def extract_and_project_datasets(h5_file_path, output_dir):
    """Extract and project datasets using LUT transformations"""
    
    # Define base images and their corresponding LUT datasets
    BASE_IMAGES = ['IMG_MIR', 'IMG_SWIR', 'IMG_TIR1', 'IMG_TIR2', 'IMG_VIS', 'IMG_WV']
    LUT_MAPPINGS = {
        'IMG_MIR': ['IMG_MIR_RADIANCE'],
        'IMG_SWIR': ['IMG_SWIR_RADIANCE'],
        'IMG_TIR1': ['IMG_TIR1_RADIANCE', 'IMG_TIR1_TEMP'],
        'IMG_TIR2': ['IMG_TIR2_RADIANCE', 'IMG_TIR2_TEMP'],
        'IMG_VIS': ['IMG_VIS_RADIANCE', 'IMG_VIS_ALBEDO'],
        'IMG_WV': ['IMG_WV_RADIANCE']
    }

    # Define projection parameters
    proj_params = {
        'proj': 'merc',
        'lon_0': 77.25,
        'lat_ts': 17.75,
        'x_0': 0,
        'y_0': 0,
        'a': 6378137,
        'b': 6356752.3142,
        'units': 'm'
    }
    crs = CRS.from_dict(proj_params)

    # Geographic bounds
    bounds = {'left': 44.5, 'right': 110.0, 'bottom': -10.0, 'top': 45.5}
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    left, bottom = transformer.transform(bounds['left'], bounds['bottom'])
    right, top = transformer.transform(bounds['right'], bounds['top'])

    with h5py.File(h5_file_path, 'r') as h5f:
        for base_image in BASE_IMAGES:
            try:
                if base_image not in h5f:
                    logger.warning(f"Base image {base_image} not found")
                    continue

                # Get base image data
                data = h5f[base_image][:]
                if len(data.shape) > 2:
                    data = np.squeeze(data)  # Remove single-dimensional entries

                # Process each corresponding LUT dataset
                for lut_dataset in LUT_MAPPINGS[base_image]:
                    try:
                        logger.info(f"Processing {lut_dataset}")
                        
                        # Get LUT data
                        if lut_dataset not in h5f:
                            logger.warning(f"LUT dataset {lut_dataset} not found")
                            continue
                            
                        lut = h5f[lut_dataset][:]
                        
                        # Apply LUT transformation
                        transformed_data = apply_lut_transformation(data, lut)

                        # Apply scale and offset if available
                        scale_factor = h5f[lut_dataset].attrs.get('scale_factor', 1.0)
                        add_offset = h5f[lut_dataset].attrs.get('add_offset', 0.0)
                        transformed_data = transformed_data * scale_factor + add_offset

                        # Modify output paths to include _cog suffix
                        temp_initial = os.path.join(output_dir, f"{lut_dataset}_initial.tif")
                        temp_3857 = os.path.join(output_dir, f"{lut_dataset}_3857.tif")
                        final_cog = os.path.join(output_dir, f"{lut_dataset}_cog.tif")

                        # Create geotransform
                        transform = from_bounds(
                            left, bottom, right, top,
                            transformed_data.shape[1], transformed_data.shape[0]
                        )

                        # Write initial GeoTIFF
                        with rasterio.open(
                            temp_initial,
                            'w',
                            driver='GTiff',
                            height=transformed_data.shape[0],
                            width=transformed_data.shape[1],
                            count=1,
                            dtype=transformed_data.dtype,
                            crs=crs.to_wkt(),
                            transform=transform,
                            nodata=None,
                            tiled=True,
                            blockxsize=512,
                            blockysize=512,
                            compress='LZW'
                        ) as dst:
                            dst.write(transformed_data, 1)
                            metadata = {
                                'wavelength': h5f[lut_dataset].attrs.get('central_wavelength', ''),
                                'units': h5f[lut_dataset].attrs.get('units', ''),
                                'scale_factor': str(scale_factor),
                                'add_offset': str(add_offset)
                            }
                            dst.update_tags(**metadata)

                        # Reproject to EPSG:3857
                        reproject_to_epsg3857(temp_initial, temp_3857)

                        # Convert to COG
                        if convert_to_cog(temp_3857, final_cog):
                            logger.info(f"Successfully created COG: {final_cog}")
                        else:
                            logger.error(f"Failed to create COG for {lut_dataset}")

                    except Exception as e:
                        logger.error(f"Error processing {lut_dataset}: {str(e)}")
                        continue

            except Exception as e:
                logger.error(f"Error processing base image {base_image}: {str(e)}")
                continue

if __name__ == "__main__":
    h5_file = "3RIMG_04SEP2024_1015_L1C_ASIA_MER_V01R00.h5"
    output_dir = "projected_data"
    extract_and_project_datasets(h5_file, output_dir)
