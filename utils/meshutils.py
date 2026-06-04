import logging

import numpy as np
from numpy.typing import NDArray
from os import makedirs
import vtk
from vtk.util import numpy_support

from utils.config import SAVE, SEED

def _create_vtk_mesh_vectorized(noise_map: NDArray[np.floating]) -> vtk.vtkPolyData:
    """
    Converts a 2D heightmap into a 3D VTK mesh using true zero-copy vectorized operations.

    Memory Layout Alignment:
    - NumPy C-contiguous arrays vary the column index fastest.
    - VTK StructuredGrids mandate the X-coordinate varies fastest.
    - Therefore, NumPy columns map to VTK X, and NumPy rows map to VTK Z.
    - The noise values map to the VTK Y-axis (elevation).
    Args:
        noise_map (NDArray[np.floating]): A 2D array representing the terrain heightmap.

    Returns:
        vtk.vtkPolyData: The generated 3D geometry compatible with VTK rendering and storage.
    """
    rows, cols = noise_map.shape
    
    # Match spatial axes to VTK's required memory layout (X varies fastest)
    x = np.arange(0, cols, 1)  # X maps to columns
    z = np.arange(0, rows, 1)  # Z maps to rows
    
    # indexing='xy' ensures mx varies fastest (cols) and mz varies slowest (rows)
    mx, mz = np.meshgrid(x, z, indexing='xy') 
    
    # Pre-allocate the (N, 3) coordinate array to avoid column_stack RAM churn
    # VTK strictly requires float32 or float64 geometry arrays
    coords = np.empty((rows * cols, 3), dtype=np.float32)

    # Write directly into the contiguous block
    coords[:, 0] = mx.ravel()
    coords[:, 1] = noise_map.ravel()
    coords[:, 2] = mz.ravel()

    # True zero-copy conversion from the structured NumPy array to VTK
    vtk_float_array = numpy_support.numpy_to_vtk(
        num_array=coords, 
        deep=False, 
        array_type=vtk.VTK_FLOAT
    )
    # vtk_float_array automatically infers 3 components from the (N, 3) shape
    
    points = vtk.vtkPoints()
    points.SetData(vtk_float_array)
    
    sgrid = vtk.vtkStructuredGrid()
    # VTK Dimensions: (X-size, Y-size, Z-size).
    # We mapped cols->X, rows->Y (in 2D grid logic, though physically Z), 1->Z
    sgrid.SetDimensions(cols, rows, 1) 
    sgrid.SetPoints(points)

    geom_filter = vtk.vtkGeometryFilter()
    geom_filter.SetInputData(sgrid)
    geom_filter.Update()
    
    return geom_filter.GetOutput()

def _save_vtk_mesh(polydata: vtk.vtkPolyData, filename: str) -> None:
    """
    Writes the provided VTK PolyData to a file in binary format.

    This utility handles the serialization of the mesh geometry to disk. 
    It forces the binary file type to ensure smaller file sizes and faster 
    loading times compared to ASCII.

    Args:
        polydata (vtk.vtkPolyData): The 3D mesh data to save.
        filename (str): The destination file path (usually ending in .vtk).
    """
    # Write the mesh data to disk
    writer: vtk.vtkPolyDataWriter = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    # Binary is preferred over ASCII for smaller file sizes and faster loading
    writer.SetFileTypeToBinary()
    writer.Write()

def gen_mesh(noise_map: NDArray[np.floating]) -> None:
    """
    Orchestrates the generation and storage of the 3D terrain mesh.

    This is the main entry point for the meshing pipeline. It accepts a final 
    noise map, converts it into a VTK mesh, and (if enabled by the configuration) 
    saves the result to the 'meshes' directory with a unique identifier.

    Args:
        noise_map (NDArray[np.floating]): The final processed heightmap to be meshed.
    """
    logging.info("Generating VTK mesh...")

    # Convert the 2D heightmap array into 3D geometry
    vtk_mesh = _create_vtk_mesh_vectorized(noise_map)
    if SAVE:
        makedirs("meshes", exist_ok=True)
        _save_vtk_mesh(vtk_mesh, f"meshes/mesh_{SEED}.vtk") 
    logging.info("VTK meshes generated")
