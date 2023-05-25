# Nucleus Feature Extraction

## How to use

### Manual install

- Install the GPU version of libtorch from [here](https://pytorch.org/get-started/locally/).
- Set the LIBTORCH environment variable to the path of the libtorch installation.
- run `nuclei-feature-extraction --help` to see the available options.
- run `nuclei-feature-extraction [options] <input-geojson> <input-slide> <output-file> <feature-set>`
- \[alternatively\] run `run.sh <input-geojson-folder> <input-slide-folder> <output-file-folder>` to process all the geojson files in the folder. (The options are set in the script for a machine with 2 nvidia A100 and 40cores). You might need to change the options in the script to match your machine.

### Docker 

```sh
docker run -v "<INPUT SLIDE FOLDER>:/in-slides -v "<INPUT GEOJSON FODLER>":/in-geojson -v <OUTPUT FODLER:/out -e EXTRACTION_FEATURES=<FEATURE SET> -e EXTRACTION_OUT_EXT=<OUTPUT FILE FORMAT> oxabz/nuclei-feature-extraction:v0.1
```