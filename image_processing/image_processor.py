import os
import json
from PIL import Image
from shutil import copyfile
from . import operations
from . import captioning

class ImageProcessor:

    def __init__(self, path, out_path, models, args):

        self.path = path
        self.out_path = out_path

        self.format = args.format
        self.quality = args.quality

        self.sd_version = args.sd_version

        self.models = models

        self.append_captions = args.append_captions
    
    def open_image(self):

        self.image = Image.open(self.path).convert('RGB')
        self.width, self.height = self.image.size
 
    def convert(self, mode):

        # mode: The mode to convert to. Must be a string like 'RGB', 'RGBA', 'L', etc.
        self.image = self.image.convert(mode)

    def crop_by_percentage(self, amount):
        
        self.image = operations.crop_by_percentage(self.image, amount)

    def save_caption(self):

        if not os.path.exists(os.path.dirname(self.out_path)):

            os.mkdir(os.path.dirname(self.out_path))  

        caption_path = f'{os.path.splitext(self.out_path)[0]}.txt'

        if self.append_captions:
            mode = 'a'
            prepend = ', '
        else:
            mode = 'w'
            prepend = ''

        with open(caption_path, mode) as f:

            # REMINDER: this checks what captions have been created, maybe confusing to write it this way?
            if hasattr(self, 'caption_blip'):
                f.write(f'{prepend}{self.caption_blip[0]}, ')
            if hasattr(self, 'caption_clip'):
                f.write(f'{prepend}{self.caption_clip}, ')
            if hasattr(self, 'caption_metadata'):
                f.write(f'{prepend}{self.caption_metadata}, ')
            if hasattr(self, 'caption_directory'):
                f.write(f'{prepend}{self.caption_directory}, ')
            # Note: maybe useful to implement checking for double captions like so:
            # if self.caption_blip[0] not in f:

    def metadata_caption(self):

        # Note: this is for the dataset from contemporary art daily that contains metadata
        metadata_path = os.path.join(os.path.dirname(self.path), 'metadata.json')

        if os.path.exists(metadata_path):
            
            f = open(metadata_path, 'r') 

            data = json.load(f)

            if not data['group_show'] and len(data['artist']) > 1:

                self.caption_metadata = f"in the style of {data['artist']}"
                print(self.caption_metadata) 

    def blip_caption(self):

        self.caption_blip = captioning.caption_image(self.image, self.models.blip_model, self.models.vis_processors)

    def blip_sort_by_questions(self):

        self.answer = captioning.sort_by_questions(self.image, self.models.blip_model, self.models.vis_processors, self.models.txt_processors)

    def blip_sort_into_folders(self):

        self.answer = captioning.sort_into_folders(self.image, self.models.blip_model, self.models.vis_processors, self.models.txt_processors)

    def interrogate_clip(self):

        self.caption_clip = captioning.interrogate_clip(self.image, self.models.ci)

    def copy_metadata(self):

        metadata_in = os.path.join(os.path.dirname(self.path), 'metadata.json')
        metadata_out = os.path.join(os.path.dirname(self.out_path), 'metadata.json')

        if os.path.exists(os.path.dirname(self.out_path)) and os.path.isfile(metadata_in) and not os.path.isfile(metadata_out):
            copyfile(metadata_in, metadata_out)

    def copy_captions(self):

        root_in, ext_in = os.path.splitext(self.path)
        root_out, ext_out = os.path.splitext(self.out_path)  

        captions_in = root_in + '.txt'
        captions_out = root_out + '.txt'

        print(captions_in, captions_out)

        if os.path.isfile(captions_in) and not os.path.isfile(captions_out):
            copyfile(captions_in, captions_out)
        
    def use_directory_name(self):

        directory_path = os.path.dirname(self.path)
        self.caption_directory = f"in the style of {str(os.path.basename(directory_path))}"

    def save_image(self):
        """
        Save the image to a file.

        Args:
            path: The path to save the image. If not specified, overwrites the original image.
            format: The format to use for the saved image. Defaults to 'JPEG'.
            quality: The quality to use for the saved image (only applicable for some formats). Defaults to 100.
        """
        
        # if the image wasn't opened, e.g. in text only mode, skip saving
        if not hasattr(self, 'image'):

            pass 

        else:
            # If the image mode is not 'RGB' and the output format is 'JPEG', convert the image to 'RGB' mode
            if self.image.mode != 'RGB' and format.upper() == 'JPEG':

                self.image = self.image.convert('RGB')

            # overwrite in place if output_dir is not given
            if self.out_path is None:

                self.out_path = self.path
            
            if not os.path.exists(os.path.dirname(self.out_path)):

                os.mkdir(os.path.dirname(self.out_path))      

            self.image.save(self.out_path, format=self.format, quality=self.quality)
