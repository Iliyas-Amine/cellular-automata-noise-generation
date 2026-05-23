import logging

import numpy as np
from numpy.typing import NDArray
from os import makedirs
import vtk
from vtk.util import numpy_support

from utils.config import SAVE, SEED

def _create_vtk_mesh_vectorized(noise_map: NDArray[np.floating]) -> vtk.vtkPolyData:
    """
    Converts a 2D heightmap into a 3D VTK mesh using efficient vectorized operations.

    This function generates a structured grid where the X and Y coordinates correspond 
    to the array indices and the Z coordinate corresponds to the noise value (height).
    It leverages NumPy's `meshgrid` and `column_stack` to prepare the data for VTK 
    without explicit Python loops, significantly improving performance.

    Args:
        noise_map (NDArray[np.floating]): A 2D array representing the terrain heightmap.

    Returns:
        vtk.vtkPolyData: The generated 3D geometry compatible with VTK rendering and storage.
    """
    rows: int
    cols: int
    rows, cols = noise_map.shape
    
    # Generate index arrays for the grid coordinates
    x: NDArray[np.intp] = np.arange(0, rows, 1)
    y: NDArray[np.intp] = np.arange(0, cols, 1)
    
    # Create 2D coordinate matrices from the 1D index arrays
    mx: NDArray[np.intp]
    my: NDArray[np.intp]
    mx, my = np.meshgrid(x, y, indexing='ij') 
    
    # Flatten everything to 1D arrays to prepare for interleaved stacking
    flat_x: NDArray[np.intp] = mx.ravel()
    flat_y: NDArray[np.floating] = noise_map.ravel() # The height (Z) comes from our noise map
    flat_z: NDArray[np.intp] = my.ravel()        
    
    # Interleave arrays to create a standard (x, y, z) points list: [x0, y0, z0, x1, y1, z1...]
    # VTK requires float32 or float64 for coordinates usually
    coords: NDArray[np.floating] = np.column_stack((flat_x, flat_y, flat_z)).ravel().astype(np.float32)

    # Zero-copy conversion from NumPy array to VTK array
    vtk_float_array: vtk.vtkFloatArray = numpy_support.numpy_to_vtk(
        num_array=coords, 
        deep=False, 
        array_type=vtk.VTK_FLOAT
    )
    # Tell VTK that every 3 values represent one point (tuple)
    vtk_float_array.SetNumberOfComponents(3)
    
    # Create the points container
    points: vtk.vtkPoints = vtk.vtkPoints()
    points.SetData(vtk_float_array)
    
    # Define the topology: StructuredGrid implies a regular grid connection between points
    sgrid: vtk.vtkStructuredGrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(cols, rows, 1) 
    sgrid.SetPoints(points)

    # Convert the StructuredGrid to PolyData (general geometry) for easier export
    geom_filter: vtk.vtkGeometryFilter = vtk.vtkGeometryFilter()
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
    logging.debug(f"Mesh#{SEED} - Noise loaded")
    
    logging.debug(f"Mesh#{SEED} - Creating primary mesh (Vectorized)...")
    # Convert the 2D heightmap array into 3D geometry
    vtk_mesh: vtk.vtkPolyData = _create_vtk_mesh_vectorized(noise_map)
    logging.debug(f"Mesh#{SEED} - Created primary mesh")
    
    if SAVE:
        logging.debug(f"Mesh#{SEED} - Saving...")
        makedirs("meshes", exist_ok=True)
        _save_vtk_mesh(vtk_mesh, f"meshes/mesh_{SEED}.vtk")
        logging.debug(f"Mesh#{SEED} - Saved")
    
    logging.info("VTK meshes generated")
