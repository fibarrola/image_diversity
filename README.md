# MEASURING DIVERSITY OF IMAGE SETS

Assess a set of image's diversity by computing an approximation of the Entropy of the images' mappings into a pre-trained network latents.

<br>

## Installation

Install from [pypi](https://pypi.org/project/image-diversity/):

```
pip install image-diversity
```

<br>

## Usage (console)

To compute the Truncated CLIP Entropy (TCE) of a set of images in a directory, run
```
python3 -m image_diversity <path/to/dir>
```

<br>

To compute the Truncated Inception Entropy (TIE), change the div_type option to TIE 
```
python3 -m image_diversity <path/to/dir> --div_type TIE
```

### Arguments

```
python3 -m image_diversity <path/to/dir>
--div_type <str>        Encoding network: TCE (default) or TIE
--n_eigs <int>          Number of eigenvalues for truncation (default: 20)
--device <str>          Device to use. Automatic by default
--batch_size <int>      Batch size for data loading (default: 16)
```
<br>

## Usage (script)

Computing Truncated CLIP Entropy (TCE)
```
from image_diversity import ClipMetrics

clip_metrics = ClipMetrics()
tce = clip_metrics.tce("path/to/img/dir")
```
<br>

Computing Truncated Inception Entropy (TIE)
```
from image_diversity import InceptionMetrics

inception_metrics = InceptionMetrics()
tie = inception_metrics.tie("path/to/img/dir")
```
<br>
