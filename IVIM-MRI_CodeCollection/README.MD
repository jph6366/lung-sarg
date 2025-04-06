# IVIM_code_collection

The ISMRM Open Science Initiative for Perfusion Imaging (OSIPI) is an initiative of the ISMRM Perfusion Study Group, founded in 2019 after a membership survey, and with a mission to:

“promote the sharing of perfusion imaging software in order to eliminate the practice of duplicate development, improve the reproducibility of perfusion imaging research, and speed up the translation into tools for discovery science, drug development, and clinical practice”

This IVIM code collection code library is maintained by OSIPI Taskforce 2.4 (currently not available) and aims to collect, test and share open-source code related to intravoxel incoherent motion (IVIM) analysis of diffusion encoded MRI data to be used in research and software development. Code contributions can include any code related IVIM analysis (denoising, motion correction, model fitting, etc.), but at an initial phase, development of tests and other features of the repository will predominantly focus on fitting algorithms. A goal of the IVIM OSIPI task force is to develop a fully tested and harmonized code library, building upon the contributions obtained through this initiative. Documentation and analysis are available on the OSIPI TF2.4.

**Table of Contents**

- [Installation](#installation)
- [Introduction](#introduction)
- [Development](#development)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

```console
pip install ....
```

## Introduction


### Main functionality
1. **OsipiBase class**: This class acts as a parent class for standardised versions of different code submissions.
    - Due to code submissions having different authors, it is expected that they all vary in their usage, inputs, and outputs. In order to facilitate testing in a larger scale, a common interface.
    - The standardized folder contains the standardised version of each code submission. Here, a class is created following a naming convention 
    
    (`<initials>_<institution>_<algorithm name>`),
    
    -  with `__init__()` and `ivim_fit()` methods that integrate well with the OsipiBase class. The idea is that every submitted fitting algorithm should be initialised in the same way, and executed in the same way.

    1. The `__init__()` method ensures that the algorithm is initiated correctly in accordance with OsipiBase.
    
    2. The purpose of `ivim_fit()` is to take a singe voxel signal and b-values as input, and return IVIM parameters as output.

### Pipeline steps:
1. Read image
 a. Read
 b. Put into canonical orientation
 c. Intensity preprocess (rescale to 0-1)
2. Segment cartilage using deep learning  
3. Register patient image to atlas image.  
4. Transform patient mesh into atlas space using registration transform  
5. Split the mesh into inner and outer surfaces in atlas space
  a. Island filtering (keep largest + islands close to atlas mesh)
  b. Mesh smoothing
  c. Clustering into inner and outer surface
6. Transform the inner and outer meshes back to patient space  
7. Measuring the thickness by computing distances between inner and outer surfaces
8. Transfer thicknesses from patient mesh to atlas mesh by iterating over atlas vertices and finding the closest patient vertex
9. Project atlas mesh to 2D using precomputed mapping

## Citation

## Acknowledgements

## License

This is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
