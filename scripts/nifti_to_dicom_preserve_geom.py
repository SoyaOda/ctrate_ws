#!/usr/bin/env python3
"""
NIfTI to DICOM conversion with preserved geometry
Based on SimpleITK official examples for proper IOP/IPP handling
"""
import json
import os
import pathlib
import SimpleITK as sitk
import argparse
import numpy as np
from datetime import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset
import pydicom.uid

def write_dicom_series_from_nifti(nifti_path, out_dir):
    """
    Convert NIfTI to DICOM series preserving full geometry
    Using pydicom for proper DICOM file structure
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading NIfTI: {nifti_path}")
    img = sitk.ReadImage(str(nifti_path))
    
    # Get numpy array from SimpleITK image
    img_array = sitk.GetArrayFromImage(img)  # (z, y, x)
    
    spacing = img.GetSpacing()      # (sx, sy, sz)
    origin = img.GetOrigin()        # (ox, oy, oz)
    direction = img.GetDirection()  # 3x3 matrix flattened to length-9 tuple
    size = img.GetSize()            # (Nx, Ny, Nz)
    
    print(f"[INFO] Size: {size}, Spacing: {spacing}")
    print(f"[INFO] Origin: {origin}")
    print(f"[INFO] Direction matrix preserved from original NIfTI")
    
    # IOP: Direction cosines for row and column directions
    iop = [
        direction[0], direction[3], direction[6],  # Row direction (X)
        direction[1], direction[4], direction[7]   # Column direction (Y)
    ]
    
    # Generate consistent UIDs
    study_uid = pydicom.uid.generate_uid()
    series_uid = pydicom.uid.generate_uid()
    frame_of_ref_uid = pydicom.uid.generate_uid()
    
    file_names = []
    meta_list = []
    
    print(f"[INFO] Writing {size[2]} DICOM slices...")
    
    for k in range(size[2]):
        # Calculate IPP (Image Position Patient) for k-th slice
        # Use SimpleITK's TransformIndexToPhysicalPoint
        ipp = img.TransformIndexToPhysicalPoint([0, 0, k])
        
        # Get slice data (need to flip from ZYX to XY for DICOM)
        slice_data = img_array[k, :, :].astype(np.int16)
        
        # Create DICOM file
        filename = str(out_dir / f"slice_{k:04d}.dcm")
        
        # File meta info
        file_meta = pydicom.Dataset()
        file_meta.FileMetaInformationGroupLength = 196
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.ImplementationVersionName = 'PYDICOM'
        
        # Create the FileDataset
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b'\x00' * 128)
        
        # Patient Module
        ds.PatientName = 'Anonymous'
        ds.PatientID = 'CTRATE001'
        ds.PatientBirthDate = ''
        ds.PatientSex = ''
        
        # General Study Module
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S.%f")
        ds.StudyID = '1'
        ds.AccessionNumber = ''
        ds.ReferringPhysicianName = ''
        
        # General Series Module
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 1
        ds.Modality = 'CT'
        ds.SeriesDescription = 'NIfTI to DICOM conversion'
        ds.BodyPartExamined = ''
        
        # Frame of Reference Module
        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.PositionReferenceIndicator = ''
        
        # General Image Module
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = list(ipp)
        ds.ImageOrientationPatient = iop
        
        # Image Pixel Module
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # [row, col]
        
        # CT Image Module
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.SliceThickness = float(spacing[2])
        ds.SpacingBetweenSlices = float(spacing[2])
        
        # SOP Common Module
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # Pixel Data
        ds.PixelData = slice_data.tobytes()
        
        # Save the file
        ds.save_as(filename)
        
        file_names.append(filename)
        meta_list.append({
            "index": k,
            "InstanceNumber": k+1,
            "IPP": list(ipp),
            "filename": filename
        })
    
    # Save slice mapping with full geometry information
    geometry_info = {
        "nifti_source": str(nifti_path),
        "dicom_output": str(out_dir),
        "files": file_names,
        "meta": meta_list,
        "geometry": {
            "origin": list(origin),
            "spacing": list(spacing),
            "direction": list(direction),
            "size": list(size),
            "iop": iop
        },
        "uids": {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "frame_of_ref_uid": frame_of_ref_uid
        }
    }
    
    # Save JSON in parent directory to avoid HARTA reading it
    json_path = out_dir.parent / f"{out_dir.name}_slice_map.json"
    with open(json_path, "w") as f:
        json.dump(geometry_info, f, indent=2)
    
    print(f"[OK] Converted {size[2]} slices to DICOM")
    print(f"[OK] Geometry preserved in: {json_path}")
    
    return json_path

def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI to DICOM with preserved geometry"
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Input NIfTI file path")
    parser.add_argument("-o", "--outdir", required=True,
                        help="Output DICOM directory")
    
    args = parser.parse_args()
    
    try:
        write_dicom_series_from_nifti(args.input, args.outdir)
    except Exception as e:
        print(f"[ERROR] Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()