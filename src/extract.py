import cv2
import argparse
from image_align import SIFTAligner
from data_extraction import MorphDataExtractor

def extract_data(im, aligner, extractor):
    query = cv2.imread(im)
    aligned_sift = aligner.align(query)

    entities = extractor.extract(aligned_sift)

    return entities


def main(args):
    query_image_location = args.query
    template_locataion = args.template
    roi_json_file = args.roi

    # roi_data = "/Users/jaswantjonnada/Documents/TestRequestForm/roi_data/TRF-1_annotation.json"

    morph_extractor = MorphDataExtractor(roi_json_file)

    # lifecell_template_path = "/Users/jaswantjonnada/Documents/TestRequestForm/templates/TRF-1.png"

    sift_aligner = SIFTAligner(template_locataion)

    entities = extract_data(im=query_image_location,
                            aligner=sift_aligner,
                            extractor=morph_extractor)
    
    print(entities)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Parsing arguments for information extraction")
    
    parser.add_argument('--query', type=str,
                    help='Image from where the data needs to be extracted')
    
    parser.add_argument('--template', type=str,
                        help="The template document which is used for alignment")
    
    parser.add_argument("--roi", type=str,
                        help="json file that contains the roi data")
    
    args = parser.parse_args()
    
    main(args)