import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from image_processing.image_processor import ImageProcessor
from image_processing.captioning import Models
import time

def process_image(image_path, out_path, models, args):

    image_operations = [
        args.blip_caption, 
        args.blip_question, 
        args.blip_folder_sort, 
        args.interrogate_clip,
        args.save_images
        ]
    skip_image_processing =  not any(image_operations) 

    if args.skip_existing_files and os.path.isfile(out_path):

        print(f"Output file already exists, skipping process for: {image_path}")
        return
    
    processor = ImageProcessor(image_path, out_path, models, args)

    # image ops
    if not skip_image_processing:

        try:
            processor.open_image()
        except Exception as e:
            print(f'An error occurred while processing {image_path}: {e}')
            return

    # text ops
    if args.blip_caption:
        processor.blip_caption()
    # TO DO: the blip sorting functions should be somewhere else and work with variables instead of hardcoding   
    if args.blip_question:
        processor.blip_sort_by_questions()
        if processor.answer == 'yes':
            processor.save()
    if args.blip_folder_sort:
        processor.blip_sort_into_folders()
        directory, filename = os.path.split(processor.out_path)
        if processor.answer == 'room':
            processor.out_path = os.path.join(directory, 'room', filename) 
        elif processor.answer == 'wall':
            processor.out_path = os.path.join(directory, 'wall', filename)
        elif processor.answer == 'none':
            processor.out_path = os.path.join(directory, 'none', filename)
        if not os.path.exists(os.path.dirname(processor.out_path)):
            os.mkdir(os.path.dirname(processor.out_path))  
    if args.interrogate_clip:
        processor.interrogate_clip()
    if args.metadata_caption:
        processor.metadata_caption()
    if args.use_directory_name:
        processor.use_directory_name()
    if args.save_captions:
        processor.save_caption()
    if args.copy_metadata:
        processor.copy_metadata()
    if args.copy_captions:
        processor.copy_captions()

    # Save the edited image
    if args.save_images:
        processor.save_image()

def main():

    start = time.time()

    parser = argparse.ArgumentParser(description='Apply image processing operations to an image or a folder of images.')
    parser.add_argument('path', type=str, help='The path to the image file or folder.')
    parser.add_argument('--output_dir', type=str, default='output', help='The path to a folder where to save the edited image(s) and caption(s).')
    parser.add_argument('--single_output_dir', action='store_true', help='If not enabled, program copies existing folder structure. Enable if single folder output is desired')
    parser.add_argument('--format', type=str, default='JPEG', help='The format to use for the saved image(s).')
    parser.add_argument('--quality', type=int, default=100, help='The quality to use for the saved image(s).')
    parser.add_argument('--text_only', action='store_true', help='skips opening and processing images')
    parser.add_argument('--blip_caption', action='store_true', help='generate a caption for the image')
    parser.add_argument('--blip_question', action='store_true', help='Answer a question about the image')
    parser.add_argument('--blip_folder_sort', action='store_true', help='Sort into folders using BLIP questions.')
    parser.add_argument('--interrogate_clip', action='store_true', help='Use CLIP interrogator to generate a caption for the image')
    parser.add_argument('--save_captions', action='store_true', help='Saves the caption as a .txt next to the image.')
    parser.add_argument('--save_images', action='store_true', help='saves the images to the output folder')
    parser.add_argument('--copy_metadata', action='store_true', help='copies metadata.json file from the input folder if it exists.')
    parser.add_argument('--copy_captions', action='store_true', help='copies filename.txt file from the input folder if it exists.')
    parser.add_argument('--append_captions', action='store_true', help='appends rather than overwrites if there are already captions in the output directory')
    parser.add_argument('--metadata_caption', action='store_true', help='Append artist name to caption if there is a metadata.json in the directory.')
    parser.add_argument('--use_directory_name', action='store_true', help="appends 'in the style of [folder name]' to the caption")
    parser.add_argument('--skip_existing_files', action='store_true', help='skips processing file if output file already exists.')
    parser.add_argument('--sd_version', type=int, default=1, help='Which version of stable diffusion to optimize for, mainly for CLIP interrogator. Can be 1 or 2')
    parser.add_argument('--threadpool', action='store_true', help='can be used to accelerate non-ai functions.')
    args = parser.parse_args()
    print(args)

    models = Models(args)

    if args.threadpool and os.path.isdir(args.path):

        with ThreadPoolExecutor() as executor:
            for dirpath, dirnames, filenames in os.walk(args.path):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(dirpath, filename)
                        relative_path = os.path.relpath(image_path, args.path)
                        if args.single_output_dir:
                            output_path = os.path.join(args.output_dir, filename)
                        else:
                            output_path = os.path.join(args.output_dir, relative_path)
                        executor.submit(process_image, image_path, output_path, models, args)

    elif os.path.isdir(args.path):

        for dirpath, dirnames, filenames in os.walk(args.path):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):           
                    image_path = os.path.join(dirpath, filename)
                    print(f'processing {image_path}')
                    relative_path = os.path.relpath(image_path, args.path)
                    if args.single_output_dir:
                        output_path = os.path.join(args.output_dir, filename)
                    else: 
                        output_path = os.path.join(args.output_dir, relative_path)
                    process_image(image_path, output_path, models, args)
    else:
        # Process a single image
        process_image(args.path, os.path.join(args.output_dir, os.path.basename(args.path)), models, args)

    end = time.time()
    total_time = end - start
    print(f"\n time taken: {str(round(total_time, 3))} seconds")

if __name__ == "__main__":
    main()