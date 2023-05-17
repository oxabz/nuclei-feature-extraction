#!/bin/bash
# Description: Run the program to extract features
# Usage: ./run.sh <geojson_folder> <slide_folder> <output_folder> [output_file_extension]

function usage {
    echo "Usage: $0 <geojson_folder> <slide_folder> <output_folder> [output_file_extension]"
    echo "  <geojson_folder>        Folder containing geojson files"
    echo "  <slide_folder>          Folder containing slide files"
    echo "  <output_folder>         Folder to store output files"
    echo "  [output_file_extension] File extension of output files (default: csv)"
    exit 1
}

geojson_folder=$1
slide_folder=$2
output_folder=$3
output_file_extension=$4

if [ -z "$geojson_folder" ]
then
    usage
fi

if [ -z "$slide_folder" ]
then
    usage
fi

if [ -z "$output_folder" ]
then
    usage
fi

if [ -z "$output_file_extension" ]
then
    output_file_extension="csv"
fi

# Create output folder if it does not exist
mkdir -p $output_folder

# Run the program
for geojson_file in $geojson_folder/*.geojson
do
    slide_file=$slide_folder/$(basename $geojson_file .geojson).svs
    output_file=$output_folder/$(basename $geojson_file .geojson).$output_file_extension
    echo "Processing $geojson_file"
    ./nuclei-feature-extraction $EXTRACTION_ARGS $geojson_file $slide_file $output_file $EXTRACTION_FEATURES
    if [ $? -ne 0 ]
    then
        echo "Error processing $geojson_file"
    else 
        echo "Done processing $geojson_file"
    fi
done
